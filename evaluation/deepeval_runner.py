"""DeepEval evaluation runner for LLM output evaluation."""

from __future__ import annotations

import os
from typing import Any, Callable

from dotenv import load_dotenv

# Load environment variables BEFORE importing deepeval to ensure API keys are available
load_dotenv()

from deepeval import evaluate  # noqa: E402
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, GEval, BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402

from config.settings import get_settings  # noqa: E402
from evaluation.rate_limiter import get_rate_limiter  # noqa: E402
from models.schemas import EvaluationResult, EvaluationScore  # noqa: E402

# Indentation for nested log output (matches phoenix_eval.py)
LOG_INDENT = "    "


def _handle_metric_error(
    e: Exception,
    metric_name: str,
    display_name: str,
    threshold: float,
    zero_div_reason: str,
    scores: list[Any],
) -> None:
    """Handle metric evaluation errors, with special handling for ZeroDivisionError.
    
    DeepEval's internal async workers can raise ZeroDivisionError when they fail
    to extract statements from the output. This helper provides consistent error
    handling across all metrics.
    """
    from models.schemas import EvaluationScore
    
    error_msg = str(e)
    is_zero_div = isinstance(e, ZeroDivisionError) or "ZeroDivisionError" in error_msg
    
    if is_zero_div:
        print(f"{LOG_INDENT}[DeepEval] {display_name} skipped: {zero_div_reason}")
        reason = zero_div_reason
    else:
        print(f"{LOG_INDENT}[DeepEval] {display_name} failed: {e}")
        reason = f"Evaluation error: {e}"
    
    scores.append(
        EvaluationScore(
            metric_name=metric_name,
            score=0.0,
            passed=False,
            threshold=threshold,
            reason=reason,
        )
    )


class ToolCorrectnessMetric(BaseMetric):
    """Custom metric to verify correct tools were called."""

    def __init__(
        self,
        expected_tools: list[str] | None = None,
        actual_tools_called: list[str] | None = None,
        threshold: float = 1.0,
    ) -> None:
        self.expected_tools = expected_tools or []
        self.actual_tools_called = actual_tools_called or []
        self.threshold = threshold
        self._score: float = 0.0
        self._reason: str = ""

    @property
    def score(self) -> float:  # type: ignore[override]
        return self._score

    @property
    def reason(self) -> str:  # type: ignore[override]
        return self._reason

    @property
    def __name__(self) -> str:
        return "ToolCorrectnessMetric"

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Measure if correct tools were called."""
        if not self.expected_tools:
            self._score = 1.0
            self._reason = "No specific tools expected."
            return self._score

        if not self.actual_tools_called:
            self._score = 0.0
            self._reason = f"No tools were called. Expected: {', '.join(self.expected_tools)}"
            return self._score

        tools_found = []
        for tool in self.expected_tools:
            if tool in self.actual_tools_called:
                tools_found.append(tool)

        if len(tools_found) == len(self.expected_tools):
            self._score = 1.0
            self._reason = f"All expected tools called: {', '.join(tools_found)}"
        elif tools_found:
            self._score = len(tools_found) / len(self.expected_tools)
            missing = [t for t in self.expected_tools if t not in tools_found]
            self._reason = f"Partial tools called. Missing: {', '.join(missing)}. Called: {', '.join(self.actual_tools_called)}"
        else:
            self._score = 0.0
            self._reason = f"Expected tools not called. Expected: {', '.join(self.expected_tools)}, Actual: {', '.join(self.actual_tools_called)}"

        return self._score

    def is_successful(self) -> bool:
        return self._score >= self.threshold


class SchemaValidationMetric(BaseMetric):
    """Custom metric to verify output schema validity."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self._score: float = 0.0
        self._reason: str = ""

    @property
    def score(self) -> float:  # type: ignore[override]
        return self._score

    @property
    def reason(self) -> str:  # type: ignore[override]
        return self._reason

    @property
    def __name__(self) -> str:
        return "SchemaValidationMetric"

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Verify output follows expected schema."""
        actual_output = test_case.actual_output or ""

        required_fields = ["type", "query"]
        found_fields = []

        for field in required_fields:
            if f'"{field}"' in actual_output or f"'{field}'" in actual_output or f"{field}:" in actual_output:
                found_fields.append(field)

        if len(found_fields) == len(required_fields):
            self._score = 1.0
            self._reason = "All required schema fields present."
        elif found_fields:
            self._score = len(found_fields) / len(required_fields)
            missing = [f for f in required_fields if f not in found_fields]
            self._reason = f"Partial schema validation. Missing: {', '.join(missing)}"
        else:
            self._score = 0.5
            self._reason = "Could not validate schema structure. Output may still be valid."

        return self._score

    def is_successful(self) -> bool:
        return self._score >= self.threshold


class DeepEvalRunner:
    """Runner for DeepEval-based evaluation of LLM outputs."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._rate_limiter = get_rate_limiter()
        self._model = self._get_eval_model()

    def _get_eval_model(self) -> str | None:
        """Get the model string for DeepEval metrics.

        DeepEval uses OPENAI_API_KEY from environment by default.
        Returns the model name to use, or None to use default.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            import warnings
            warnings.warn(
                "OPENAI_API_KEY not found in environment. "
                "DeepEval metrics require an OpenAI API key for LLM-as-judge evaluation."
            )
            return None
        return "gpt-4o-mini"

    async def _run_metric(
        self,
        metric: Any,
        test_case: LLMTestCase,
        metric_name: str,
        display_name: str,
        threshold: float,
        zero_div_reason: str,
        scores: list[EvaluationScore],
        rate_limit_name: str | None = None,
        score_transform: Callable[[float], float] | None = None,
        reason_transform: Callable[[str, float, float], str] | None = None,
    ) -> None:
        """Run a single DeepEval metric with rate limiting and error handling."""
        print(f"{LOG_INDENT}[DeepEval] Running {display_name.lower()} eval...")
        try:
            if rate_limit_name:
                await self._rate_limiter.execute_with_retry(
                    metric.measure, test_case,
                    service_name="DeepEval", metric_name=rate_limit_name,
                )
            else:
                metric.measure(test_case)
            
            raw_score = metric.score if metric.score is not None else 0.0
            final_score = score_transform(raw_score) if score_transform else raw_score
            
            reason = metric.reason or ""
            if reason_transform and metric.score is not None:
                reason = reason_transform(reason, raw_score, final_score)
            
            scores.append(EvaluationScore(
                metric_name=metric_name,
                score=final_score,
                passed=metric.is_successful(),
                threshold=threshold,
                reason=reason,
            ))
            print(f"{LOG_INDENT}[DeepEval] {display_name} complete: {final_score:.2f}")
        except Exception as e:
            _handle_metric_error(
                e, metric_name=metric_name, display_name=display_name,
                threshold=threshold, zero_div_reason=zero_div_reason, scores=scores,
            )

    async def evaluate(
        self,
        test_case_id: str,
        input_query: str,
        actual_output: str,
        context: list[str] | None = None,
        expected_output: str | None = None,
        expected_tools: list[str] | None = None,
        actual_tools_called: list[str] | None = None,
    ) -> list[EvaluationScore]:
        """
        Evaluate a single test case using DeepEval metrics.

        Args:
            test_case_id: Unique identifier for the test case
            input_query: The input query
            actual_output: The actual output from the agent
            context: Context/retrieval context for evaluation
            expected_output: Expected output for comparison
            expected_tools: Tools expected to be called
            actual_tools_called: Tools that were actually called by the agent

        Returns:
            List of EvaluationScore objects
        """
        context = context or []
        actual_tools_called = actual_tools_called or []
        scores: list[EvaluationScore] = []

        test_case = LLMTestCase(
            input=input_query,
            actual_output=actual_output,
            expected_output=expected_output or "",
            context=context,  # For HallucinationMetric
            retrieval_context=context,  # For FaithfulnessMetric
        )

        await self._run_metric(
            AnswerRelevancyMetric(threshold=0.7, model=self._model, strict_mode=False),
            test_case, "deepeval_answer_relevancy", "Answer relevancy", 0.7,
            "Could not extract statements from output for relevancy evaluation",
            scores, rate_limit_name="answer_relevancy",
        )

        if expected_output:
            correctness_metric = GEval(
                name="Correctness",
                evaluation_steps=[
                    "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
                    "Heavily penalize omission of important details",
                    "Vague language or contradicting opinions are acceptable",
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                model=self._model, threshold=0.7,
            )
            await self._run_metric(
                correctness_metric, test_case, "deepeval_correctness", "Correctness", 0.7,
                "Could not evaluate correctness - internal evaluation error",
                scores, rate_limit_name="correctness",
            )

        if context:
            await self._run_metric(
                FaithfulnessMetric(threshold=0.8, model=self._model),
                test_case, "deepeval_faithfulness", "Faithfulness", 0.8,
                "Could not extract claims from output for faithfulness evaluation",
                scores, rate_limit_name="faithfulness",
            )

            def _invert_score(raw: float) -> float:
                return 1.0 - raw

            def _adjust_hallucination_reason(reason: str, raw: float, final: float) -> str:
                return reason.replace(f"score is {raw:.2f}", f"score is {final:.2f}")

            await self._run_metric(
                HallucinationMetric(threshold=0.5, model=self._model),
                test_case, "deepeval_hallucination", "Hallucination", 0.5,
                "Could not evaluate hallucination - no valid contexts",
                scores, rate_limit_name="hallucination",
                score_transform=_invert_score, reason_transform=_adjust_hallucination_reason,
            )

        if expected_tools:
            await self._run_metric(
                ToolCorrectnessMetric(expected_tools=expected_tools, actual_tools_called=actual_tools_called, threshold=1.0),
                test_case, "deepeval_tool_correctness", "Tool correctness", 1.0,
                "Could not evaluate tool correctness", scores,
            )

        await self._run_metric(
            SchemaValidationMetric(threshold=0.8),
            test_case, "deepeval_schema_validation", "Schema validation", 0.8,
            "Could not validate schema", scores,
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
                - expected_tools (optional)
                - actual_tools_called (optional)

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
                expected_tools=tc.get("expected_tools"),
                actual_tools_called=tc.get("actual_tools_called"),
            )

            overall_passed = all(s.passed for s in scores)

            result = EvaluationResult(
                test_case_id=tc["test_case_id"],
                input_query=tc["input_query"],
                actual_output=tc["actual_output"],
                expected_output=tc.get("expected_output"),
                context=tc.get("context", []),
                api_results=tc.get("api_results", []),
                deepeval_scores=scores,
                phoenix_scores=[],
                overall_passed=overall_passed,
            )
            results.append(result)

        return results

    def run_cli_tests(self, test_cases: list[LLMTestCase]) -> None:
        """Run tests via DeepEval CLI-compatible interface."""
        metrics = [
            AnswerRelevancyMetric(threshold=0.7, model=self._model),
        ]

        evaluate(test_cases=test_cases, metrics=metrics)  # type: ignore[operator]
