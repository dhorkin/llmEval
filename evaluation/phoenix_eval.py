"""Phoenix evaluation runner for LLM output evaluation."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv

# Load environment variables BEFORE importing phoenix to ensure API keys are available
load_dotenv()

import nest_asyncio
import pandas as pd
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.evals.models import OpenAIModel

from config.settings import get_settings
from models.schemas import EvaluationResult, EvaluationScore

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


class PhoenixEvaluator:
    """Evaluator using Arize Phoenix for LLM output quality assessment."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: OpenAIModel | None = None
        self._evaluators_initialized = False
        self._hallucination_eval: HallucinationEvaluator | None = None
        self._qa_eval: QAEvaluator | None = None
        self._relevance_eval: RelevanceEvaluator | None = None

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
        """Initialize Phoenix evaluators."""
        if self._evaluators_initialized:
            return

        model = self._get_model()
        self._hallucination_eval = HallucinationEvaluator(model)
        self._qa_eval = QAEvaluator(model)
        self._relevance_eval = RelevanceEvaluator(model)
        self._evaluators_initialized = True

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

        df = pd.DataFrame(
            {
                "input": [input_query],
                "output": [actual_output],
                "reference": [context_str],
            }
        )

        scores: list[EvaluationScore] = []

        try:
            print(f"{LOG_INDENT}[Phoenix] Running hallucination eval...")
            with _indented_output():
                hallucination_results = run_evals(
                    dataframe=df,
                    evaluators=[self._hallucination_eval],
                    provide_explanation=True,
                    concurrency=1,
                )
            if not hallucination_results[0].empty:
                row = hallucination_results[0].iloc[0]
                label = _safe_get_value(row, "label", "")
                explanation = _safe_get_value(row, "explanation", "No explanation provided")
                score = 1.0 if label == "factual" else 0.0
                scores.append(
                    EvaluationScore(
                        metric_name="phoenix_hallucination",
                        score=score,
                        passed=score >= 0.8,
                        threshold=0.8,
                        reason=explanation if explanation else f"Label: {label}",
                    )
                )
            else:
                scores.append(
                    EvaluationScore(
                        metric_name="phoenix_hallucination",
                        score=0.0,
                        passed=False,
                        threshold=0.8,
                        reason="Evaluation returned empty results",
                    )
                )
            print(f"{LOG_INDENT}[Phoenix] Hallucination eval complete")
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

        try:
            print(f"{LOG_INDENT}[Phoenix] Running QA correctness eval...")
            qa_df = pd.DataFrame(
                {
                    "input": [input_query],
                    "output": [actual_output],
                    "reference": [expected_output or actual_output],
                }
            )
            with _indented_output():
                qa_results = run_evals(
                    dataframe=qa_df,
                    evaluators=[self._qa_eval],
                    provide_explanation=True,
                    concurrency=1,
                )
            if not qa_results[0].empty:
                row = qa_results[0].iloc[0]
                label = _safe_get_value(row, "label", "")
                explanation = _safe_get_value(row, "explanation", "No explanation provided")
                score = 1.0 if label == "correct" else 0.0
                scores.append(
                    EvaluationScore(
                        metric_name="phoenix_qa_correctness",
                        score=score,
                        passed=score >= 0.7,
                        threshold=0.7,
                        reason=explanation if explanation else f"Label: {label}",
                    )
                )
            else:
                scores.append(
                    EvaluationScore(
                        metric_name="phoenix_qa_correctness",
                        score=0.0,
                        passed=False,
                        threshold=0.7,
                        reason="Evaluation returned empty results",
                    )
                )
            print(f"{LOG_INDENT}[Phoenix] QA correctness eval complete")
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

        try:
            print(f"{LOG_INDENT}[Phoenix] Running relevance eval...")
            # For relevance, use the query + expected output as reference context
            # This evaluates whether the output is relevant to answering the query
            relevance_reference = input_query
            if expected_output:
                relevance_reference = f"Query: {input_query}\nExpected: {expected_output}"
            elif context_str:
                relevance_reference = f"Query: {input_query}\nContext: {context_str}"
            
            relevance_df = pd.DataFrame(
                {
                    "input": [input_query],
                    "output": [actual_output],
                    "reference": [relevance_reference],
                }
            )
            with _indented_output():
                relevance_results = run_evals(
                    dataframe=relevance_df,
                    evaluators=[self._relevance_eval],
                    provide_explanation=True,
                    concurrency=1,
                )
            if not relevance_results[0].empty:
                row = relevance_results[0].iloc[0]
                label = _safe_get_value(row, "label", "")
                explanation = _safe_get_value(row, "explanation", "No explanation provided")
                score = 1.0 if label == "relevant" else 0.0
                scores.append(
                    EvaluationScore(
                        metric_name="phoenix_relevance",
                        score=score,
                        passed=score >= 0.7,
                        threshold=0.7,
                        reason=explanation if explanation else f"Label: {label}",
                    )
                )
            else:
                scores.append(
                    EvaluationScore(
                        metric_name="phoenix_relevance",
                        score=0.0,
                        passed=False,
                        threshold=0.7,
                        reason="Evaluation returned empty results",
                    )
                )
            print(f"{LOG_INDENT}[Phoenix] Relevance eval complete")
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
                deepeval_scores=[],
                phoenix_scores=scores,
                overall_passed=overall_passed,
            )
            results.append(result)

        return results
