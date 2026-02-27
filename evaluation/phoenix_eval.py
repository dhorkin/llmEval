"""Phoenix evaluation runner for LLM output evaluation."""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from typing import Any, Literal

from dotenv import load_dotenv

# Load environment variables BEFORE importing phoenix to ensure API keys are available
load_dotenv()

import nest_asyncio
import pandas as pd
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    llm_classify,
    run_evals,
)
from phoenix.evals.models import OpenAIModel

from config.settings import get_settings
from models.schemas import EvaluationResult, EvaluationScore

PhoenixEvalMethod = Literal["categorical", "discrete", "continuous"]

# Apply nest_asyncio to suppress Phoenix's notebook warning and enable proper async handling
nest_asyncio.apply()

# Indentation for nested log output
LOG_INDENT = "    "


class _IndentedWriter:
    """Wrapper that adds indentation prefix to each line of output."""
    
    # Extra indent for progress bars (they are sub-tasks)
    PROGRESS_BAR_EXTRA_INDENT = "  "
    
    def __init__(self, original, prefix: str):
        self._original = original
        self._prefix = prefix
        self._progress_prefix = prefix + self.PROGRESS_BAR_EXTRA_INDENT
        self._at_line_start = True
    
    def _shorten_progress_bar(self, text: str, shorten_by: int) -> str:
        """Shorten tqdm progress bar by removing characters from the bar portion."""
        import re
        # Match progress bar pattern: |████████████████| or similar
        pattern = r"\|([█░▒▓ ]+)\|"
        match = re.search(pattern, text)
        if match:
            bar = match.group(1)
            if len(bar) > shorten_by + 10:
                # Remove characters from the bar
                new_len = len(bar) - shorten_by
                new_bar = bar[:new_len]
                text = text[:match.start(1)] + new_bar + text[match.end(1):]
        return text
    
    def _is_progress_bar(self, text: str) -> bool:
        """Check if text is a tqdm progress bar line."""
        # Match common Phoenix progress bar prefixes
        progress_prefixes = ("run_evals", "llm_classify", "evaluate")
        return "|" in text and any(p in text for p in progress_prefixes)
    
    def write(self, text: str) -> int:
        if not text:
            return 0
        
        # Check if this is a progress bar line
        is_progress_bar = self._is_progress_bar(text)
        
        if is_progress_bar:
            # Suppress initial 0% progress lines
            if "0/1 (0.0%)" in text or "(0.0%)" in text:
                return 0
            # Shorten progress bar to compensate for extra indentation
            text = self._shorten_progress_bar(text, len(self._progress_prefix))
            prefix = self._progress_prefix
        else:
            prefix = self._prefix
        
        # Handle carriage return (used by tqdm for progress bar updates)
        if "\r" in text:
            text = text.replace("\r", f"\r{prefix}")
            if not text.startswith(prefix):
                text = prefix + text
        elif self._at_line_start:
            text = prefix + text
        self._at_line_start = text.endswith("\n")
        return self._original.write(text)
    
    def flush(self):
        self._original.flush()
    
    def __getattr__(self, name):
        return getattr(self._original, name)


@contextmanager
def _indented_output(prefix: str = LOG_INDENT):
    """Context manager that adds indentation to stdout/stderr output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _IndentedWriter(old_stdout, prefix)
    sys.stderr = _IndentedWriter(old_stderr, prefix)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _safe_get_value(series: pd.Series, key: str, default: Any = None) -> Any:
    """Safely get a value from a pandas Series, handling NaN values."""
    try:
        value = series.get(key, default)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


# Discrete evaluation templates (5-point scale)
DISCRETE_HALLUCINATION_TEMPLATE = """You are evaluating whether an AI response contains hallucinations or fabricated information.

Input Query: {input}
AI Response: {output}
Reference Context: {reference}

Rate the factual accuracy of the response on a 5-point scale:
- excellent: Response is completely factual with no hallucinations
- good: Response is mostly factual with minor unsupported claims
- fair: Response has some factual content but also some unsupported claims
- poor: Response contains significant hallucinations or fabrications
- very_poor: Response is mostly or entirely fabricated

Provide your rating and a one-sentence explanation."""

DISCRETE_QA_TEMPLATE = """You are evaluating the correctness of an AI response to a question.

Question: {input}
AI Response: {output}
Expected Answer: {reference}

Rate the correctness of the response on a 5-point scale:
- excellent: Response fully and accurately answers the question
- good: Response mostly answers the question with minor gaps
- fair: Response partially answers the question
- poor: Response barely addresses the question or has significant errors
- very_poor: Response does not answer the question or is completely wrong

Provide your rating and a one-sentence explanation."""

DISCRETE_RELEVANCE_TEMPLATE = """You are evaluating whether an AI response is relevant to the input query.

Input Query: {input}
AI Response: {output}
Context: {reference}

Rate the relevance of the response on a 5-point scale:
- excellent: Response is highly relevant and directly addresses the query
- good: Response is relevant with minor tangential content
- fair: Response is somewhat relevant but includes irrelevant content
- poor: Response is mostly irrelevant to the query
- very_poor: Response is completely off-topic

Provide your rating and a one-sentence explanation."""

DISCRETE_SCALE_MAP = {
    "excellent": 1.0,
    "good": 0.75,
    "fair": 0.5,
    "poor": 0.25,
    "very_poor": 0.0,
}

# Continuous evaluation templates (0.00-1.00 scale)
CONTINUOUS_HALLUCINATION_TEMPLATE = """You are evaluating whether an AI response contains hallucinations or fabricated information.

Input Query: {input}
AI Response: {output}
Reference Context: {reference}

Evaluate the factual accuracy of the response and provide a score from 0.00 to 1.00 where:
- 1.00 = Completely factual, no hallucinations
- 0.75 = Mostly factual with minor unsupported claims
- 0.50 = Mixed factual and fabricated content
- 0.25 = Significant hallucinations
- 0.00 = Completely fabricated

Respond with JSON in this exact format:
{{"score": <number between 0.00 and 1.00>, "explanation": "<one sentence>"}}"""

CONTINUOUS_QA_TEMPLATE = """You are evaluating the correctness of an AI response to a question.

Question: {input}
AI Response: {output}
Expected Answer: {reference}

Evaluate the correctness of the response and provide a score from 0.00 to 1.00 where:
- 1.00 = Fully correct and complete answer
- 0.75 = Mostly correct with minor gaps
- 0.50 = Partially correct
- 0.25 = Barely correct with significant errors
- 0.00 = Completely incorrect

Respond with JSON in this exact format:
{{"score": <number between 0.00 and 1.00>, "explanation": "<one sentence>"}}"""

CONTINUOUS_RELEVANCE_TEMPLATE = """You are evaluating whether an AI response is relevant to the input query.

Input Query: {input}
AI Response: {output}
Context: {reference}

Evaluate the relevance of the response and provide a score from 0.00 to 1.00 where:
- 1.00 = Highly relevant, directly addresses the query
- 0.75 = Relevant with minor tangential content
- 0.50 = Somewhat relevant
- 0.25 = Mostly irrelevant
- 0.00 = Completely off-topic

Respond with JSON in this exact format:
{{"score": <number between 0.00 and 1.00>, "explanation": "<one sentence>"}}"""


def _parse_continuous_response(response: str) -> tuple[float, str]:
    """Parse a continuous evaluation response to extract score and explanation."""
    try:
        # Try to parse as JSON first
        data = json.loads(response)
        score = float(data.get("score", 0.0))
        explanation = data.get("explanation", "No explanation provided")
        return round(score, 2), explanation
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: try to extract score from text
    score_match = re.search(r"(?:score|rating)[:\s]*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
        score = max(0.0, min(1.0, score))
        return round(score, 2), response

    return 0.0, f"Could not parse response: {response[:200]}"


class PhoenixEvaluator:
    """Evaluator using Arize Phoenix for LLM output quality assessment."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: OpenAIModel | None = None
        self._evaluators_initialized = False
        self._hallucination_eval: HallucinationEvaluator | None = None
        self._qa_eval: QAEvaluator | None = None
        self._relevance_eval: RelevanceEvaluator | None = None
        self._eval_method: PhoenixEvalMethod = self.settings.phoenix_evaluation_method

    def _get_model(self) -> OpenAIModel:
        """Get or create OpenAI model for evaluation."""
        if self._model is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                import warnings
                warnings.warn(
                    "OPENAI_API_KEY not found. Phoenix evaluators require an OpenAI API key."
                )
            self._model = OpenAIModel(
                model="gpt-4o-mini",
                api_key=api_key,
                initial_rate_limit=int(self.settings.eval_rate_limit_initial_rps),
            )
        return self._model

    def _init_evaluators(self) -> None:
        """Initialize Phoenix evaluators (only for categorical mode)."""
        if self._evaluators_initialized:
            return

        model = self._get_model()
        if self._eval_method == "categorical":
            self._hallucination_eval = HallucinationEvaluator(model)
            self._qa_eval = QAEvaluator(model)
            self._relevance_eval = RelevanceEvaluator(model)
        self._evaluators_initialized = True

    def _evaluate_discrete(
        self,
        df: pd.DataFrame,
        template: str,
        metric_name: str,
    ) -> tuple[float, str]:
        """Evaluate using discrete 5-point scale via llm_classify."""
        model = self._get_model()
        rails = list(DISCRETE_SCALE_MAP.keys())

        with _indented_output():
            results = llm_classify(
                dataframe=df,
                model=model,
                template=template,
                rails=rails,
                provide_explanation=True,
                concurrency=1,
            )

        if results.empty:
            return 0.0, "Evaluation returned empty results"

        row = results.iloc[0]
        label = _safe_get_value(row, "label", "")
        explanation = _safe_get_value(row, "explanation", "No explanation provided")
        score = DISCRETE_SCALE_MAP.get(label, 0.0)
        return score, explanation or f"Label: {label}"

    def _evaluate_continuous(
        self,
        input_query: str,
        output: str,
        reference: str,
        template: str,
    ) -> tuple[float, str]:
        """Evaluate using continuous 0.00-1.00 scale via direct LLM call."""
        model = self._get_model()
        prompt = template.format(input=input_query, output=output, reference=reference)

        try:
            response = model(prompt)
            score, explanation = _parse_continuous_response(response)
            return score, explanation
        except Exception as e:
            return 0.0, f"Evaluation error: {e}"

    async def evaluate(
        self,
        test_case_id: str,
        input_query: str,
        actual_output: str,
        context: list[str] | None = None,
        expected_output: str | None = None,
    ) -> list[EvaluationScore]:
        """
        Evaluate a single test case using Phoenix evaluators.

        Args:
            test_case_id: Unique identifier for the test case
            input_query: The input query
            actual_output: The actual output from the agent
            context: Context/retrieval context for evaluation
            expected_output: Expected output for comparison

        Returns:
            List of EvaluationScore objects
        """
        self._init_evaluators()
        context = context or []
        context_str = "\n".join(context) if context else ""

        print(f"{LOG_INDENT}[Phoenix] Using {self._eval_method} evaluation mode")
        scores: list[EvaluationScore] = []

        # Build reference strings
        hallucination_reference = context_str
        qa_reference = expected_output or actual_output
        relevance_reference = input_query
        if expected_output:
            relevance_reference = f"Query: {input_query}\nExpected: {expected_output}"
        elif context_str:
            relevance_reference = f"Query: {input_query}\nContext: {context_str}"

        # Create dataframes for evaluation
        hallucination_df = pd.DataFrame({
            "input": [input_query],
            "output": [actual_output],
            "reference": [hallucination_reference],
        })
        qa_df = pd.DataFrame({
            "input": [input_query],
            "output": [actual_output],
            "reference": [qa_reference],
        })
        relevance_df = pd.DataFrame({
            "input": [input_query],
            "output": [actual_output],
            "reference": [relevance_reference],
        })

        # Hallucination evaluation
        try:
            print(f"{LOG_INDENT}[Phoenix] Running hallucination eval...")
            if self._eval_method == "categorical":
                score, explanation = self._evaluate_categorical_hallucination(hallucination_df)
            elif self._eval_method == "discrete":
                score, explanation = self._evaluate_discrete(
                    hallucination_df, DISCRETE_HALLUCINATION_TEMPLATE, "hallucination"
                )
            else:  # continuous
                score, explanation = self._evaluate_continuous(
                    input_query, actual_output, hallucination_reference,
                    CONTINUOUS_HALLUCINATION_TEMPLATE
                )
            scores.append(
                EvaluationScore(
                    metric_name="phoenix_hallucination",
                    score=score,
                    passed=score >= 0.8,
                    threshold=0.8,
                    reason=explanation,
                )
            )
            print(f"{LOG_INDENT}[Phoenix] Hallucination eval complete: {score}")
        except Exception as e:
            print(f"{LOG_INDENT}[Phoenix] Hallucination eval failed: {e}")
            scores.append(
                EvaluationScore(
                    metric_name="phoenix_hallucination",
                    score=0.0,
                    passed=False,
                    threshold=0.8,
                    reason=f"Evaluation error: {e}",
                )
            )

        # QA Correctness evaluation
        try:
            print(f"{LOG_INDENT}[Phoenix] Running QA correctness eval...")
            if self._eval_method == "categorical":
                score, explanation = self._evaluate_categorical_qa(qa_df)
            elif self._eval_method == "discrete":
                score, explanation = self._evaluate_discrete(
                    qa_df, DISCRETE_QA_TEMPLATE, "qa_correctness"
                )
            else:  # continuous
                score, explanation = self._evaluate_continuous(
                    input_query, actual_output, qa_reference,
                    CONTINUOUS_QA_TEMPLATE
                )
            scores.append(
                EvaluationScore(
                    metric_name="phoenix_qa_correctness",
                    score=score,
                    passed=score >= 0.7,
                    threshold=0.7,
                    reason=explanation,
                )
            )
            print(f"{LOG_INDENT}[Phoenix] QA correctness eval complete: {score}")
        except Exception as e:
            print(f"{LOG_INDENT}[Phoenix] QA correctness eval failed: {e}")
            scores.append(
                EvaluationScore(
                    metric_name="phoenix_qa_correctness",
                    score=0.0,
                    passed=False,
                    threshold=0.7,
                    reason=f"Evaluation error: {e}",
                )
            )

        # Relevance evaluation
        try:
            print(f"{LOG_INDENT}[Phoenix] Running relevance eval...")
            if self._eval_method == "categorical":
                score, explanation = self._evaluate_categorical_relevance(relevance_df)
            elif self._eval_method == "discrete":
                score, explanation = self._evaluate_discrete(
                    relevance_df, DISCRETE_RELEVANCE_TEMPLATE, "relevance"
                )
            else:  # continuous
                score, explanation = self._evaluate_continuous(
                    input_query, actual_output, relevance_reference,
                    CONTINUOUS_RELEVANCE_TEMPLATE
                )
            scores.append(
                EvaluationScore(
                    metric_name="phoenix_relevance",
                    score=score,
                    passed=score >= 0.7,
                    threshold=0.7,
                    reason=explanation,
                )
            )
            print(f"{LOG_INDENT}[Phoenix] Relevance eval complete: {score}")
        except Exception as e:
            print(f"{LOG_INDENT}[Phoenix] Relevance eval failed: {e}")
            scores.append(
                EvaluationScore(
                    metric_name="phoenix_relevance",
                    score=0.0,
                    passed=False,
                    threshold=0.7,
                    reason=f"Evaluation error: {e}",
                )
            )

        return scores

    def _evaluate_categorical_hallucination(self, df: pd.DataFrame) -> tuple[float, str]:
        """Evaluate hallucination using categorical (binary) mode."""
        with _indented_output():
            results = run_evals(
                dataframe=df,
                evaluators=[self._hallucination_eval],
                provide_explanation=True,
                concurrency=1,
            )
        if results[0].empty:
            return 0.0, "Evaluation returned empty results"
        row = results[0].iloc[0]
        label = _safe_get_value(row, "label", "")
        explanation = _safe_get_value(row, "explanation", "No explanation provided")
        score = 1.0 if label == "factual" else 0.0
        return score, explanation or f"Label: {label}"

    def _evaluate_categorical_qa(self, df: pd.DataFrame) -> tuple[float, str]:
        """Evaluate QA correctness using categorical (binary) mode."""
        with _indented_output():
            results = run_evals(
                dataframe=df,
                evaluators=[self._qa_eval],
                provide_explanation=True,
                concurrency=1,
            )
        if results[0].empty:
            return 0.0, "Evaluation returned empty results"
        row = results[0].iloc[0]
        label = _safe_get_value(row, "label", "")
        explanation = _safe_get_value(row, "explanation", "No explanation provided")
        score = 1.0 if label == "correct" else 0.0
        return score, explanation or f"Label: {label}"

    def _evaluate_categorical_relevance(self, df: pd.DataFrame) -> tuple[float, str]:
        """Evaluate relevance using categorical (binary) mode."""
        with _indented_output():
            results = run_evals(
                dataframe=df,
                evaluators=[self._relevance_eval],
                provide_explanation=True,
                concurrency=1,
            )
        if results[0].empty:
            return 0.0, "Evaluation returned empty results"
        row = results[0].iloc[0]
        label = _safe_get_value(row, "label", "")
        explanation = _safe_get_value(row, "explanation", "No explanation provided")
        score = 1.0 if label == "relevant" else 0.0
        return score, explanation or f"Label: {label}"

    async def evaluate_batch(
        self,
        test_cases: list[dict[str, Any]],
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of test case dictionaries with keys:
                - test_case_id
                - input_query
                - actual_output
                - context (optional)
                - expected_output (optional)

        Returns:
            List of EvaluationResult objects
        """
        results: list[EvaluationResult] = []

        for tc in test_cases:
            scores = await self.evaluate(
                test_case_id=tc["test_case_id"],
                input_query=tc["input_query"],
                actual_output=tc["actual_output"],
                context=tc.get("context"),
                expected_output=tc.get("expected_output"),
            )

            overall_passed = all(s.passed for s in scores)

            result = EvaluationResult(
                test_case_id=tc["test_case_id"],
                input_query=tc["input_query"],
                actual_output=tc["actual_output"],
                expected_output=tc.get("expected_output"),
                context=tc.get("context", []),
                api_results=tc.get("api_results", []),
                deepeval_scores=[],
                phoenix_scores=scores,
                overall_passed=overall_passed,
            )
            results.append(result)

        return results
