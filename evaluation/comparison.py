"""Comparison module for evaluating with both Phoenix and DeepEval."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from evaluation.deepeval_runner import DeepEvalRunner
from evaluation.phoenix_eval import PhoenixEvaluator
from models.schemas import EvaluationResult, EvaluationScore


class EvaluationComparison:
    """Run both Phoenix and DeepEval evaluations and compare results."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.phoenix_evaluator = PhoenixEvaluator()
        self.deepeval_runner = DeepEvalRunner()
        self.console = Console()

    async def evaluate_with_both(
        self,
        test_case_id: str,
        input_query: str,
        actual_output: str,
        context: list[str] | None = None,
        expected_output: str | None = None,
        expected_tools: list[str] | None = None,
        actual_tools_called: list[str] | None = None,
    ) -> EvaluationResult:
        """
        Run both Phoenix and DeepEval evaluations on a single test case.

        Returns combined EvaluationResult with scores from both frameworks.
        """
        phoenix_scores = await self.phoenix_evaluator.evaluate(
            test_case_id=test_case_id,
            input_query=input_query,
            actual_output=actual_output,
            context=context,
            expected_output=expected_output,
        )

        deepeval_scores = await self.deepeval_runner.evaluate(
            test_case_id=test_case_id,
            input_query=input_query,
            actual_output=actual_output,
            context=context,
            expected_output=expected_output,
            expected_tools=expected_tools,
            actual_tools_called=actual_tools_called,
        )

        all_scores = phoenix_scores + deepeval_scores
        overall_passed = all(s.passed for s in all_scores)

        return EvaluationResult(
            test_case_id=test_case_id,
            input_query=input_query,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context or [],
            deepeval_scores=deepeval_scores,
            phoenix_scores=phoenix_scores,
            overall_passed=overall_passed,
        )

    async def evaluate_batch_with_both(
        self,
        test_cases: list[dict[str, Any]],
        verbose: bool = True,
    ) -> list[EvaluationResult]:
        """Evaluate multiple test cases with both frameworks."""
        results: list[EvaluationResult] = []
        total = len(test_cases)

        for i, tc in enumerate(test_cases, 1):
            test_case_id = tc["test_case_id"]
            
            if verbose:
                self.console.print(
                    f"  [{i}/{total}] Evaluating [cyan]{test_case_id}[/cyan]..."
                )

            phoenix_scores = await self.phoenix_evaluator.evaluate(
                test_case_id=test_case_id,
                input_query=tc["input_query"],
                actual_output=tc["actual_output"],
                context=tc.get("context"),
                expected_output=tc.get("expected_output"),
            )
            
            if verbose:
                self.console.print(
                    f"    [green]✓[/green] Phoenix complete "
                    f"({len(phoenix_scores)} metrics)"
                )

            deepeval_scores = await self.deepeval_runner.evaluate(
                test_case_id=test_case_id,
                input_query=tc["input_query"],
                actual_output=tc["actual_output"],
                context=tc.get("context"),
                expected_output=tc.get("expected_output"),
                expected_tools=tc.get("expected_tools"),
                actual_tools_called=tc.get("actual_tools_called"),
            )
            
            if verbose:
                self.console.print(
                    f"    [green]✓[/green] DeepEval complete "
                    f"({len(deepeval_scores)} metrics)"
                )

            all_scores = phoenix_scores + deepeval_scores
            overall_passed = all(s.passed for s in all_scores)

            result = EvaluationResult(
                test_case_id=test_case_id,
                input_query=tc["input_query"],
                actual_output=tc["actual_output"],
                expected_output=tc.get("expected_output"),
                context=tc.get("context") or [],
                deepeval_scores=deepeval_scores,
                phoenix_scores=phoenix_scores,
                overall_passed=overall_passed,
            )
            results.append(result)

        if verbose and total > 0:
            self.console.print(
                f"  [bold green]✓[/bold green] All {total} test cases evaluated"
            )

        return results

    def print_comparison_report(self, results: list[EvaluationResult]) -> None:
        """Print a formatted comparison report to console."""
        self.console.print("\n[bold blue]EVALUATION COMPARISON REPORT[/bold blue]\n")

        for result in results:
            self.console.print(f"[bold]Test Case:[/bold] {result.test_case_id}")
            self.console.print(f"[dim]Query:[/dim] {result.input_query}")
            self.console.print()

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("DeepEval Score", justify="center")
            table.add_column("Phoenix Score", justify="center")
            table.add_column("Status", justify="center")

            metric_pairs = self._pair_metrics(
                result.deepeval_scores, result.phoenix_scores
            )

            for metric_name, deepeval_score, phoenix_score in metric_pairs:
                de_str = f"{deepeval_score:.2f}" if deepeval_score is not None else "-"
                ph_str = f"{phoenix_score:.2f}" if phoenix_score is not None else "-"

                if deepeval_score is not None and phoenix_score is not None:
                    diff = abs(deepeval_score - phoenix_score)
                    if diff < 0.1:
                        status = "[green]AGREE[/green]"
                    elif diff < 0.2:
                        status = "[yellow]CLOSE[/yellow]"
                    else:
                        status = "[red]DIFFER[/red]"
                else:
                    status = "[dim]N/A[/dim]"

                table.add_row(metric_name, de_str, ph_str, status)

            self.console.print(table)

            # Print brief reason summary highlighting discrepancies
            reason_summary = self._summarize_reasons(
                result.deepeval_scores, result.phoenix_scores, metric_pairs
            )
            if reason_summary:
                self.console.print(f"[dim]{reason_summary}[/dim]")

            agreement = self._calculate_agreement(
                result.deepeval_scores, result.phoenix_scores
            )
            overall_status = (
                "[green]PASSED[/green]"
                if result.overall_passed
                else "[red]FAILED[/red]"
            )

            self.console.print(
                f"Agreement: {agreement:.0%} | Overall: {overall_status}"
            )
            self.console.print("-" * 60)
            self.console.print()

    def _pair_metrics(
        self,
        deepeval_scores: list[EvaluationScore],
        phoenix_scores: list[EvaluationScore],
    ) -> list[tuple[str, float | None, float | None]]:
        """Pair similar metrics from both frameworks."""
        pairs: list[tuple[str, float | None, float | None]] = []

        metric_mapping = {
            "relevance": ("deepeval_answer_relevancy", "phoenix_relevance"),
            "faithfulness": ("deepeval_faithfulness", "phoenix_hallucination"),
            "correctness": ("deepeval_answer_relevancy", "phoenix_qa_correctness"),
            "tool_usage": ("deepeval_tool_correctness", None),
            "schema": ("deepeval_schema_validation", None),
        }

        de_scores = {s.metric_name: s.score for s in deepeval_scores}
        ph_scores = {s.metric_name: s.score for s in phoenix_scores}

        for display_name, (de_name, ph_name) in metric_mapping.items():
            de_score = de_scores.get(de_name) if de_name else None
            ph_score = ph_scores.get(ph_name) if ph_name else None

            if de_score is not None or ph_score is not None:
                pairs.append((display_name, de_score, ph_score))

        return pairs

    def _summarize_reasons(
        self,
        deepeval_scores: list[EvaluationScore],
        phoenix_scores: list[EvaluationScore],
        metric_pairs: list[tuple[str, float | None, float | None]],
    ) -> str:
        """Generate a brief summary highlighting discrepancies between frameworks."""
        de_by_name = {s.metric_name: s for s in deepeval_scores}
        ph_by_name = {s.metric_name: s for s in phoenix_scores}

        metric_mapping = {
            "relevance": ("deepeval_answer_relevancy", "phoenix_relevance"),
            "faithfulness": ("deepeval_faithfulness", "phoenix_hallucination"),
            "correctness": ("deepeval_answer_relevancy", "phoenix_qa_correctness"),
        }

        discrepancies: list[str] = []

        for display_name, de_score, ph_score in metric_pairs:
            if de_score is None or ph_score is None:
                continue
            diff = abs(de_score - ph_score)
            if diff < 0.2:
                continue

            mapping = metric_mapping.get(display_name)
            if not mapping:
                continue

            de_name, ph_name = mapping
            de_eval = de_by_name.get(de_name)
            ph_eval = ph_by_name.get(ph_name)

            if de_eval and ph_eval:
                # Determine which framework scored higher
                if de_score > ph_score:
                    higher, lower = "DeepEval", "Phoenix"
                    higher_reason = self._normalize_reason(de_eval.reason)
                    lower_reason = self._normalize_reason(ph_eval.reason)
                else:
                    higher, lower = "Phoenix", "DeepEval"
                    higher_reason = self._normalize_reason(ph_eval.reason)
                    lower_reason = self._normalize_reason(de_eval.reason)

                discrepancies.append(
                    f"Discrepancy ({display_name}): {higher} says {higher_reason}; {lower} says {lower_reason}"
                )

        if not discrepancies:
            return ""

        return "\n".join(discrepancies)

    def _normalize_reason(self, reason: str | None) -> str:
        """Return the reason with the first letter lowercased, or a default if missing."""
        if not reason:
            return "no reason"
        stripped = reason.strip()
        if not stripped:
            return "no reason"
        return stripped[0].lower() + stripped[1:]

    def _calculate_agreement(
        self,
        deepeval_scores: list[EvaluationScore],
        phoenix_scores: list[EvaluationScore],
    ) -> float:
        """Calculate agreement percentage between frameworks."""
        pairs = self._pair_metrics(deepeval_scores, phoenix_scores)
        comparable = [
            (de, ph) for _, de, ph in pairs if de is not None and ph is not None
        ]

        if not comparable:
            return 1.0

        agreements = sum(
            1 for de, ph in comparable if abs(de - ph) < 0.2
        )

        return agreements / len(comparable)

    def save_results(
        self,
        results: list[EvaluationResult],
        filename: str | None = None,
    ) -> Path:
        """Save evaluation results to JSON file in logs directory."""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"

        filepath = logs_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [r.model_dump() for r in results],
            "summary": self._generate_summary(results),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def _generate_summary(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Generate summary statistics for evaluation results."""
        total = len(results)
        passed = sum(1 for r in results if r.overall_passed)

        all_deepeval_scores: list[float] = []
        all_phoenix_scores: list[float] = []

        for r in results:
            all_deepeval_scores.extend(s.score for s in r.deepeval_scores)
            all_phoenix_scores.extend(s.score for s in r.phoenix_scores)

        return {
            "total_test_cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_deepeval_score": (
                sum(all_deepeval_scores) / len(all_deepeval_scores)
                if all_deepeval_scores
                else 0
            ),
            "avg_phoenix_score": (
                sum(all_phoenix_scores) / len(all_phoenix_scores)
                if all_phoenix_scores
                else 0
            ),
        }

    def load_results(self, filepath: str | Path) -> list[EvaluationResult]:
        """Load evaluation results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        return [EvaluationResult(**r) for r in data["results"]]

    def check_drift(
        self,
        current_results: list[EvaluationResult],
        baseline_file: str | Path | None = None,
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """
        Check for drift by comparing current results to baseline.

        Args:
            current_results: Current evaluation results
            baseline_file: Path to baseline results file
            threshold: Maximum allowed score drop (default 10%)

        Returns:
            Drift analysis report
        """
        current_summary = self._generate_summary(current_results)

        if baseline_file is None:
            logs_dir = Path("logs")
            log_files = sorted(logs_dir.glob("evaluation_*.json"))
            if len(log_files) < 2:
                return {
                    "drift_detected": False,
                    "message": "Insufficient historical data for drift detection",
                }
            baseline_file = log_files[-2]

        with open(baseline_file) as f:
            baseline_data = json.load(f)
            baseline_summary = baseline_data["summary"]

        drift_report: dict[str, Any] = {
            "drift_detected": False,
            "metrics": {},
        }

        for metric in ["avg_deepeval_score", "avg_phoenix_score", "pass_rate"]:
            current_val = current_summary.get(metric, 0)
            baseline_val = baseline_summary.get(metric, 0)
            change = current_val - baseline_val

            drift_report["metrics"][metric] = {
                "current": current_val,
                "baseline": baseline_val,
                "change": change,
                "drift": abs(change) > threshold,
            }

            if change < -threshold:
                drift_report["drift_detected"] = True

        if drift_report["drift_detected"]:
            drift_report["message"] = (
                f"DRIFT DETECTED: Scores dropped more than {threshold:.0%} from baseline"
            )
        else:
            drift_report["message"] = "No significant drift detected"

        return drift_report
