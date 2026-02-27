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
        api_results: list[dict[str, Any]] | None = None,
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
            expected_tools=expected_tools,
            actual_tools_called=actual_tools_called,
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

        # Determine pass/fail using global thresholds
        overall_passed = self._check_overall_passed(deepeval_scores, phoenix_scores)

        return EvaluationResult(
            test_case_id=test_case_id,
            input_query=input_query,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context or [],
            api_results=api_results or [],
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
                expected_tools=tc.get("expected_tools"),
                actual_tools_called=tc.get("actual_tools_called"),
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

            # Determine pass/fail using global thresholds
            overall_passed = self._check_overall_passed(deepeval_scores, phoenix_scores)

            result = EvaluationResult(
                test_case_id=test_case_id,
                input_query=tc["input_query"],
                actual_output=tc["actual_output"],
                expected_output=tc.get("expected_output"),
                context=tc.get("context") or [],
                api_results=tc.get("api_results") or [],
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

    def print_metrics_guide(self) -> None:
        """Print a guide explaining what each evaluation metric measures."""
        metric_threshold = self.settings.minimum_metric_pass_threshold
        
        self.console.print("\n[bold cyan]EVALUATION METRICS GUIDE[/bold cyan]")
        self.console.print("-" * 40)
        self.console.print(
            f"Pass threshold: [yellow]{metric_threshold}[/yellow] "
            "(configurable via MINIMUM_METRIC_PASS_THRESHOLD)\n"
        )
        
        self.console.print("Metrics (evaluated by both DeepEval and Phoenix):")
        self.console.print("  [dim]•[/dim] [bold]Relevance[/bold]: Is the output relevant to the query?")
        self.console.print("  [dim]•[/dim] [bold]Correctness[/bold]: Is the answer factually correct?")
        self.console.print("  [dim]•[/dim] [bold]Faithfulness[/bold]: Does output only contain info from context?")
        self.console.print("  [dim]•[/dim] [bold]Hallucination[/bold]: Is the output free of fabricated info?")
        self.console.print("    [italic dim]* DeepEval score is INVERTED (1.0 - raw) so higher = better[/dim italic]")
        self.console.print("  [dim]•[/dim] [bold]Tool Correctness[/bold]: Were the correct tools called?")
        self.console.print("  [dim]•[/dim] [bold]Schema[/bold]: Does output match expected schema? [dim](DeepEval only)[/dim]")
        self.console.print()
        self.console.print("[dim]All metrics use 0.0-1.0 scale where higher = better quality.[/dim]")
        self.console.print()

    def print_comparison_report(self, results: list[EvaluationResult]) -> None:
        """Print a formatted comparison report to console."""
        self.print_metrics_guide()
        
        self.console.print("[bold blue]EVALUATION COMPARISON REPORT[/bold blue]\n")
        
        agreement_threshold = self.settings.minimum_agreement_pass_threshold
        metric_threshold = self.settings.minimum_metric_pass_threshold
        eval_method = self.settings.phoenix_evaluation_method
        
        # Warn about categorical mode limitations
        if eval_method == "categorical":
            self.console.print(
                "[dim]Note: Using categorical mode (binary 0/1 scores). "
                "Agreement reflects pass/fail alignment, not score similarity.[/dim]\n"
            )

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
                    if eval_method == "categorical":
                        # For categorical: check pass/fail alignment (0.5 threshold)
                        de_pass = deepeval_score >= 0.5
                        ph_pass = phoenix_score >= 0.5
                        if de_pass == ph_pass:
                            status = "[green]AGREE[/green]"
                        else:
                            status = "[red]DIFFER[/red]"
                    else:
                        # For continuous/discrete: use score difference
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

            # Collect failed individual metrics using configurable threshold
            failed_metrics = self._get_failed_metrics(
                result.deepeval_scores, result.phoenix_scores, metric_threshold
            )
            
            agreement = self._calculate_agreement(
                result.deepeval_scores, result.phoenix_scores
            )
            
            # Determine pass/fail based on both thresholds
            agreement_passed = agreement >= agreement_threshold
            metrics_passed = len(failed_metrics) == 0
            overall_passed = agreement_passed and metrics_passed
            
            overall_status = (
                "[green]PASSED[/green]"
                if overall_passed
                else "[red]FAILED[/red]"
            )

            self.console.print(
                f"Agreement: {agreement:.0%} | Overall: {overall_status}"
            )
            
            # Show failure explanation if failed
            if not overall_passed:
                self._print_failure_explanation(
                    agreement, agreement_threshold, metric_threshold,
                    failed_metrics, agreement_passed, metrics_passed
                )
            
            self.console.print("-" * 60)
            self.console.print()

    def _pair_metrics(
        self,
        deepeval_scores: list[EvaluationScore],
        phoenix_scores: list[EvaluationScore],
    ) -> list[tuple[str, float | None, float | None]]:
        """Pair similar metrics from both frameworks.
        
        Always returns all 6 metrics, showing None (displayed as "-") when a 
        metric wasn't run by either framework.
        """
        pairs: list[tuple[str, float | None, float | None]] = []

        metric_mapping = {
            "relevance": ("deepeval_answer_relevancy", "phoenix_relevance"),
            "hallucination": ("deepeval_hallucination", "phoenix_hallucination"),
            "correctness": ("deepeval_correctness", "phoenix_qa_correctness"),
            "faithfulness": ("deepeval_faithfulness", "phoenix_faithfulness"),
            "tool_usage": ("deepeval_tool_correctness", "phoenix_tool_selection"),
            "schema": ("deepeval_schema_validation", None),
        }

        de_scores = {s.metric_name: s.score for s in deepeval_scores}
        ph_scores = {s.metric_name: s.score for s in phoenix_scores}

        for display_name, (de_name, ph_name) in metric_mapping.items():
            de_score = de_scores.get(de_name) if de_name else None
            ph_score = ph_scores.get(ph_name) if ph_name else None
            pairs.append((display_name, de_score, ph_score))

        return pairs

    def _summarize_reasons(
        self,
        deepeval_scores: list[EvaluationScore],
        phoenix_scores: list[EvaluationScore],
        metric_pairs: list[tuple[str, float | None, float | None]],
    ) -> str:
        """Generate a brief summary of discrepancies and failures.
        
        Reports:
        - Discrepancies/variances when frameworks disagree on scores
        - Metric failures even when frameworks agree or only one has the metric
        """
        de_by_name = {s.metric_name: s for s in deepeval_scores}
        ph_by_name = {s.metric_name: s for s in phoenix_scores}
        eval_method = self.settings.phoenix_evaluation_method

        metric_mapping = {
            "relevance": ("deepeval_answer_relevancy", "phoenix_relevance"),
            "hallucination": ("deepeval_hallucination", "phoenix_hallucination"),
            "correctness": ("deepeval_correctness", "phoenix_qa_correctness"),
            "faithfulness": ("deepeval_faithfulness", "phoenix_faithfulness"),
            "tool_usage": ("deepeval_tool_correctness", "phoenix_tool_selection"),
        }

        summaries: list[str] = []

        for display_name, de_score, ph_score in metric_pairs:
            mapping = metric_mapping.get(display_name)
            if not mapping:
                continue

            de_name, ph_name = mapping
            de_eval = de_by_name.get(de_name)
            ph_eval = ph_by_name.get(ph_name)

            # Check for failures (below threshold)
            de_failed = de_eval and de_score is not None and de_score < de_eval.threshold
            ph_failed = ph_eval and ph_score is not None and ph_score < ph_eval.threshold

            # Check for discrepancy between frameworks
            has_discrepancy = False
            severity = ""
            if de_score is not None and ph_score is not None:
                if eval_method == "categorical":
                    de_pass = de_score >= 0.5
                    ph_pass = ph_score >= 0.5
                    if de_pass != ph_pass:
                        has_discrepancy = True
                        severity = "Disagreement"
                else:
                    diff = abs(de_score - ph_score)
                    if diff >= 0.1:
                        has_discrepancy = True
                        severity = "Discrepancy" if diff >= 0.2 else "Variance"

            # Build summary for this metric
            if has_discrepancy and de_eval and ph_eval and de_score is not None and ph_score is not None:
                if de_score > ph_score:
                    higher, lower = "DeepEval", "Phoenix"
                    higher_reason = self._normalize_reason(de_eval.reason)
                    lower_reason = self._normalize_reason(ph_eval.reason)
                else:
                    higher, lower = "Phoenix", "DeepEval"
                    higher_reason = self._normalize_reason(ph_eval.reason)
                    lower_reason = self._normalize_reason(de_eval.reason)

                summaries.append(
                    f"{severity} ({display_name}): {higher} says {higher_reason.rstrip('.')}; {lower} says {lower_reason}"
                )
            elif de_failed or ph_failed:
                # No discrepancy but at least one failure - report failures
                parts = []
                if de_failed and de_eval:
                    reason = self._normalize_reason(de_eval.reason)
                    parts.append(f"DeepEval: {reason}")
                if ph_failed and ph_eval:
                    reason = self._normalize_reason(ph_eval.reason)
                    parts.append(f"Phoenix: {reason}")
                if parts:
                    summaries.append(f"Failed ({display_name}): {parts[0].rstrip('.')}; {parts[1]}" if len(parts) == 2 else f"Failed ({display_name}): {parts[0]}")

        return "\n".join(summaries)

    def _normalize_reason(self, reason: str | None) -> str:
        """Return the reason with the first letter lowercased, or a default if missing."""
        if not reason:
            return "no reason"
        stripped = reason.strip()
        if not stripped:
            return "no reason"
        return stripped[0].lower() + stripped[1:]

    def _get_failed_metrics(
        self,
        deepeval_scores: list[EvaluationScore],
        phoenix_scores: list[EvaluationScore],
        metric_threshold: float | None = None,
    ) -> list[tuple[str, str, float, float, str]]:
        """Get list of metrics that failed the minimum metric threshold.
        
        Args:
            deepeval_scores: DeepEval evaluation scores
            phoenix_scores: Phoenix evaluation scores
            metric_threshold: Minimum score threshold. If None, uses each metric's own threshold.
        
        Returns list of tuples: (framework, metric_name, score, threshold, reason)
        """
        failed: list[tuple[str, str, float, float, str]] = []
        threshold = metric_threshold if metric_threshold is not None else None
        
        for score in deepeval_scores:
            effective_threshold = threshold if threshold is not None else score.threshold
            if score.score < effective_threshold:
                failed.append((
                    "DeepEval",
                    score.metric_name.replace("deepeval_", ""),
                    score.score,
                    effective_threshold,
                    score.reason or "No reason provided",
                ))
        
        for score in phoenix_scores:
            effective_threshold = threshold if threshold is not None else score.threshold
            if score.score < effective_threshold:
                failed.append((
                    "Phoenix",
                    score.metric_name.replace("phoenix_", ""),
                    score.score,
                    effective_threshold,
                    score.reason or "No reason provided",
                ))
        
        return failed

    def _print_failure_explanation(
        self,
        agreement: float,
        agreement_threshold: float,
        metric_threshold: float,
        failed_metrics: list[tuple[str, str, float, float, str]],
        agreement_passed: bool,
        metrics_passed: bool,
    ) -> None:
        """Print detailed explanation of why the test case failed."""
        self.console.print()
        
        if not agreement_passed:
            self.console.print(
                f"[yellow]⚠ Agreement ({agreement:.0%}) below threshold ({agreement_threshold:.0%})[/yellow]"
            )
        
        if not metrics_passed:
            self.console.print(
                f"[yellow]⚠ Metrics below minimum threshold ({metric_threshold}):[/yellow]"
            )
            for framework, metric, score, thresh, reason in failed_metrics:
                # Truncate reason if too long
                reason_short = reason[:80] + "..." if len(reason) > 80 else reason
                self.console.print(
                    f"  [dim]• {framework} {metric}: {score:.2f} < {thresh:.2f}[/dim]"
                )
                self.console.print(f"    [dim italic]{reason_short}[/dim italic]")

    def _calculate_agreement(
        self,
        deepeval_scores: list[EvaluationScore],
        phoenix_scores: list[EvaluationScore],
    ) -> float:
        """Calculate agreement percentage between frameworks.
        
        For continuous/discrete modes: Returns average similarity (1 - normalized_diff).
        For categorical mode: Returns percentage of metrics where both frameworks
        agree on pass/fail (using 0.5 as the threshold for binary classification).
        """
        pairs = self._pair_metrics(deepeval_scores, phoenix_scores)
        comparable = [
            (de, ph) for _, de, ph in pairs if de is not None and ph is not None
        ]

        if not comparable:
            return 1.0

        eval_method = self.settings.phoenix_evaluation_method
        
        if eval_method == "categorical":
            # For categorical mode, check pass/fail alignment
            # Both scores are considered "pass" if >= 0.5, "fail" if < 0.5
            agreements = sum(
                1 for de, ph in comparable
                if (de >= 0.5) == (ph >= 0.5)
            )
            return agreements / len(comparable)
        else:
            # For continuous/discrete, use score similarity
            similarities = [1.0 - abs(de - ph) for de, ph in comparable]
            return sum(similarities) / len(comparable)

    def _check_overall_passed(
        self,
        deepeval_scores: list[EvaluationScore],
        phoenix_scores: list[EvaluationScore],
    ) -> bool:
        """Check if test case passes using global thresholds.
        
        A test case passes only if:
        1. All metrics meet MINIMUM_METRIC_PASS_THRESHOLD
        2. Framework agreement meets MINIMUM_AGREEMENT_PASS_THRESHOLD
        """
        metric_threshold = self.settings.minimum_metric_pass_threshold
        agreement_threshold = self.settings.minimum_agreement_pass_threshold
        
        # Check all metrics against global threshold
        all_scores = deepeval_scores + phoenix_scores
        metrics_passed = all(
            score.score >= metric_threshold for score in all_scores
        )
        
        # Check agreement between frameworks
        agreement = self._calculate_agreement(deepeval_scores, phoenix_scores)
        agreement_passed = agreement >= agreement_threshold
        
        return metrics_passed and agreement_passed

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
