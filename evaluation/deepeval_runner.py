"""DeepEval evaluation runner for LLM output evaluation."""

from __future__ import annotations

import os
from typing import Any

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
    scores: list,
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

        print(f"{LOG_INDENT}[DeepEval] Running answer relevancy eval...")
        try:
            relevancy_metric = AnswerRelevancyMetric(
                threshold=0.7,
                model=self._model,
                strict_mode=False,
            )
            await self._rate_limiter.execute_with_retry(
                relevancy_metric.measure,
                test_case,
                service_name="DeepEval",
                metric_name="answer_relevancy",
            )
            scores.append(
                EvaluationScore(
                    metric_name="deepeval_answer_relevancy",
                    score=relevancy_metric.score if relevancy_metric.score is not None else 0.0,
                    passed=relevancy_metric.is_successful(),
                    threshold=0.7,
                    reason=relevancy_metric.reason,
                )
            )
            print(f"{LOG_INDENT}[DeepEval] Answer relevancy complete: {relevancy_metric.score:.2f}")
        except Exception as e:
            _handle_metric_error(
                e,
                metric_name="deepeval_answer_relevancy",
                display_name="Answer relevancy",
                threshold=0.7,
                zero_div_reason="Could not extract statements from output for relevancy evaluation",
                scores=scores,
            )

        if expected_output:
            print(f"{LOG_INDENT}[DeepEval] Running correctness eval (GEval)...")
            try:
                correctness_metric = GEval(
                    name="Correctness",
                    evaluation_steps=[
                        "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
                        "Heavily penalize omission of important details",
                        "Vague language or contradicting opinions are acceptable",
                    ],
                    evaluation_params=[
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParams.EXPECTED_OUTPUT,
                    ],
                    model=self._model,
                    threshold=0.7,
                )
                await self._rate_limiter.execute_with_retry(
                    correctness_metric.measure,
                    test_case,
                    service_name="DeepEval",
                    metric_name="correctness",
                )
                scores.append(
                    EvaluationScore(
                        metric_name="deepeval_correctness",
                        score=correctness_metric.score if correctness_metric.score is not None else 0.0,
                        passed=correctness_metric.is_successful(),
                        threshold=0.7,
                        reason=correctness_metric.reason,
                    )
                )
                print(f"{LOG_INDENT}[DeepEval] Correctness complete: {correctness_metric.score:.2f}")
            except Exception as e:
                _handle_metric_error(
                    e,
                    metric_name="deepeval_correctness",
                    display_name="Correctness",
                    threshold=0.7,
                    zero_div_reason="Could not evaluate correctness - internal evaluation error",
                    scores=scores,
                )

        if context:
            print(f"{LOG_INDENT}[DeepEval] Running faithfulness eval...")
            try:
                faithfulness_metric = FaithfulnessMetric(
                    threshold=0.8,
                    model=self._model,
                )
                await self._rate_limiter.execute_with_retry(
                    faithfulness_metric.measure,
                    test_case,
                    service_name="DeepEval",
                    metric_name="faithfulness",
                )
                scores.append(
                    EvaluationScore(
                        metric_name="deepeval_faithfulness",
                        score=faithfulness_metric.score if faithfulness_metric.score is not None else 0.0,
                        passed=faithfulness_metric.is_successful(),
                        threshold=0.8,
                        reason=faithfulness_metric.reason,
                    )
                )
                print(f"{LOG_INDENT}[DeepEval] Faithfulness complete: {faithfulness_metric.score:.2f}")
            except Exception as e:
                _handle_metric_error(
                    e,
                    metric_name="deepeval_faithfulness",
                    display_name="Faithfulness",
                    threshold=0.8,
                    zero_div_reason="Could not extract claims from output for faithfulness evaluation",
                    scores=scores,
                )

            print(f"{LOG_INDENT}[DeepEval] Running hallucination eval...")
            try:
                hallucination_metric = HallucinationMetric(
                    threshold=0.5,
                    model=self._model,
                )
                await self._rate_limiter.execute_with_retry(
                    hallucination_metric.measure,
                    test_case,
                    service_name="DeepEval",
                    metric_name="hallucination",
                )
                raw_score = hallucination_metric.score if hallucination_metric.score is not None else 0.0
                hallucination_score = 1.0 - raw_score
                original_reason = hallucination_metric.reason or ""
                if hallucination_metric.score is not None:
                    original_score_str = f"{hallucination_metric.score:.2f}"
                    inverted_score_str = f"{hallucination_score:.2f}"
                    adjusted_reason = original_reason.replace(
                        f"score is {original_score_str}",
                        f"score is {inverted_score_str}"
                    )
                else:
                    adjusted_reason = original_reason
                scores.append(
                    EvaluationScore(
                        metric_name="deepeval_hallucination",
                        score=hallucination_score,
                        passed=hallucination_metric.is_successful(),
                        threshold=0.5,
                        reason=adjusted_reason,
                    )
                )
                print(f"{LOG_INDENT}[DeepEval] Hallucination complete: {hallucination_score:.2f}")
            except Exception as e:
                _handle_metric_error(
                    e,
                    metric_name="deepeval_hallucination",
                    display_name="Hallucination",
                    threshold=0.5,
                    zero_div_reason="Could not evaluate hallucination - no valid contexts",
                    scores=scores,
                )

        if expected_tools:
            print(f"{LOG_INDENT}[DeepEval] Running tool correctness eval...")
            try:
                tool_metric = ToolCorrectnessMetric(
                    expected_tools=expected_tools,
                    actual_tools_called=actual_tools_called,
                    threshold=1.0,
                )
                tool_metric.measure(test_case)
                scores.append(
                    EvaluationScore(
                        metric_name="deepeval_tool_correctness",
                        score=tool_metric.score,
                        passed=tool_metric.is_successful(),
                        threshold=1.0,
                        reason=tool_metric.reason,
                    )
                )
                print(f"{LOG_INDENT}[DeepEval] Tool correctness complete: {tool_metric.score:.2f}")
            except Exception as e:
                print(f"{LOG_INDENT}[DeepEval] Tool correctness failed: {e}")
                scores.append(
                    EvaluationScore(
                        metric_name="deepeval_tool_correctness",
                        score=0.0,
                        passed=False,
                        threshold=1.0,
                        reason=f"Evaluation error: {e}",
                    )
                )

        print(f"{LOG_INDENT}[DeepEval] Running schema validation eval...")
        try:
            schema_metric = SchemaValidationMetric(threshold=0.8)
            schema_metric.measure(test_case)
            scores.append(
                EvaluationScore(
                    metric_name="deepeval_schema_validation",
                    score=schema_metric.score,
                    passed=schema_metric.is_successful(),
                    threshold=0.8,
                    reason=schema_metric.reason,
                )
            )
            print(f"{LOG_INDENT}[DeepEval] Schema validation complete: {schema_metric.score:.2f}")
        except Exception as e:
            print(f"{LOG_INDENT}[DeepEval] Schema validation failed: {e}")
            scores.append(
                EvaluationScore(
                    metric_name="deepeval_schema_validation",
                    score=0.0,
                    passed=False,
                    threshold=0.8,
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
