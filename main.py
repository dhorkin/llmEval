"""CLI entry point for LLM Evaluation Agent."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agent.planner import AgentPlanner
from config.settings import get_settings
from evaluation.comparison import EvaluationComparison
from evaluation.deepeval_runner import DeepEvalRunner
from evaluation.phoenix_eval import PhoenixEvaluator


console = Console()

# Global state for partial result saving on interrupt
_partial_results: list[Any] = []
_agent_outputs: list[dict[str, Any]] = []
_comparison_instance: EvaluationComparison | None = None


def _handle_sigint(signum, frame):
    """Handle Ctrl+C by saving partial results and exiting immediately."""
    console.print("\n\n[yellow]Interrupted by user.[/yellow]")
    
    saved_something = False
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Save evaluation results if we have any
    if _partial_results and _comparison_instance:
        try:
            filepath = _comparison_instance.save_results(
                _partial_results, 
                filename=f"evaluation_{timestamp}_partial.json"
            )
            console.print(f"[green]Saved {len(_partial_results)} evaluation results to: {filepath}[/green]")
            saved_something = True
        except Exception as e:
            console.print(f"[red]Failed to save evaluation results: {e}[/red]")
    
    # Save agent outputs (API call results) if we have any beyond what's already evaluated
    unevaluated_outputs = _agent_outputs[len(_partial_results):]
    if unevaluated_outputs or (_agent_outputs and not _partial_results):
        outputs_to_save = _agent_outputs if not _partial_results else unevaluated_outputs
        try:
            filepath = logs_dir / f"agent_outputs_{timestamp}_partial.json"
            with open(filepath, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "status": "interrupted",
                    "total_collected": len(_agent_outputs),
                    "evaluated": len(_partial_results),
                    "agent_outputs": outputs_to_save,
                }, f, indent=2)
            console.print(f"[green]Saved {len(outputs_to_save)} agent outputs (API results) to: {filepath}[/green]")
            saved_something = True
        except Exception as e:
            console.print(f"[red]Failed to save agent outputs: {e}[/red]")
    
    if not saved_something:
        console.print("[dim]No partial results to save.[/dim]")
    
    os._exit(130)


signal.signal(signal.SIGINT, _handle_sigint)


TEST_CASES = [
    {
        "test_case_id": "book_001",
        "input_query": "Find all books written by George Orwell published before 1950",
        "expected_tools": ["book_search"],
        "expected_output": "List of Orwell books before 1950 with theme analysis",
    },
    {
        "test_case_id": "neo_001",
        "input_query": "Check if there are any Near Earth Objects passing by Earth this weekend",
        "expected_tools": ["nasa_neo"],
        "expected_output": "NEO report for the weekend with risk assessment",
    },
    {
        "test_case_id": "poetry_001",
        "input_query": "Find a sonnet by William Shakespeare and explain the metaphor used in the first quatrain",
        "expected_tools": ["poetry_search"],
        "expected_output": "Shakespeare sonnet with quatrain analysis",
    },
    {
        "test_case_id": "nutrition_001",
        "input_query": "Based on a Mediterranean diet, recommend three dinner options that avoid dairy and nuts",
        "expected_tools": ["nutrition_meal_recommendation"],
        "expected_output": "Three Mediterranean meals without dairy or nuts",
    },
    {
        "test_case_id": "edge_001",
        "input_query": "Find books by Unknown Author XYZ",
        "expected_tools": ["book_search"],
        "expected_output": "Empty result with no_results flag",
    },
    {
        "test_case_id": "failure_001",
        "input_query": "Find all books about quantum physics by Einstein published after 2020",
        "expected_tools": ["book_search"],
        "expected_output": "Empty result acknowledging Einstein died in 1955",
    },
]


@click.group()
@click.version_option(version="1.0.0")
def cli() -> None:
    """LLM Evaluation Agent - Query APIs with AI-powered analysis."""
    pass


@cli.command()
@click.argument("query")
def query(query: str) -> None:
    """Run agent with a single query."""
    console.print(Panel(f"[bold]Query:[/bold] {query}", title="LLM Agent"))

    async def run() -> None:
        agent = AgentPlanner()
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing query...", total=None)
                response = await agent.run(query)
                progress.update(task, completed=True)

            console.print("\n[bold green]Response:[/bold green]")
            console.print_json(response.report.model_dump_json(indent=2))

            console.print(f"\n[dim]Intent: {response.intent}[/dim]")
            console.print(f"[dim]Latency: {response.total_latency_ms:.0f}ms[/dim]")

            if response.tool_calls:
                console.print("\n[bold]Tool Calls:[/bold]")
                for tc in response.tool_calls:
                    status = "[green]✓[/green]" if tc.success else "[red]✗[/red]"
                    console.print(f"  {status} {tc.tool_name} ({tc.latency_ms:.0f}ms)")

        finally:
            await agent.close()

    asyncio.run(run())


@cli.command()
@click.option(
    "--framework",
    type=click.Choice(["deepeval", "phoenix", "both"]),
    default="both",
    help="Evaluation framework to use",
)
def evaluate(framework: str) -> None:
    """Run evaluation on predefined test cases."""
    global _partial_results, _comparison_instance, _agent_outputs
    
    console.print(Panel(f"Running {framework} evaluation", title="Evaluation"))

    async def run() -> None:
        global _partial_results, _comparison_instance, _agent_outputs
        
        agent = AgentPlanner()
        agent_outputs: list[dict[str, Any]] = []
        _partial_results = []
        _agent_outputs = []

        try:
            console.print("\n[bold]Generating agent outputs...[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for tc in TEST_CASES:
                    task = progress.add_task(f"Running {tc['test_case_id']}...", total=None)
                    try:
                        response = await agent.run(tc["input_query"])
                        actual_tools = [t.tool_name for t in response.tool_calls]
                        api_results = [
                            {
                                "tool_name": call.tool_name,
                                "parameters": call.parameters,
                                "result": call.result,
                                "success": call.success,
                                "error_message": call.error_message,
                                "latency_ms": call.latency_ms,
                            }
                            for call in response.tool_calls
                        ]
                        output_entry = {
                            "test_case_id": tc["test_case_id"],
                            "input_query": tc["input_query"],
                            "actual_output": response.report.model_dump_json(),
                            "expected_output": tc.get("expected_output"),
                            "expected_tools": tc.get("expected_tools"),
                            "actual_tools_called": actual_tools,
                            "context": [str(call) for call in response.tool_calls],
                            "api_results": api_results,
                        }
                        agent_outputs.append(output_entry)
                        _agent_outputs.append(output_entry)
                    except Exception as e:
                        error_entry = {
                            "test_case_id": tc["test_case_id"],
                            "input_query": tc["input_query"],
                            "actual_output": f"Error: {e}",
                            "expected_output": tc.get("expected_output"),
                            "actual_tools_called": [],
                            "api_results": [],
                        }
                        agent_outputs.append(error_entry)
                        _agent_outputs.append(error_entry)
                    progress.update(task, completed=True)

        finally:
            await agent.close()

        console.print("\n[bold]Running evaluations...[/bold]")

        if framework == "deepeval":
            runner = DeepEvalRunner()
            eval_results = await runner.evaluate_batch(agent_outputs)
            _print_eval_results(eval_results, "DeepEval")

        elif framework == "phoenix":
            evaluator = PhoenixEvaluator()
            eval_results = await evaluator.evaluate_batch(agent_outputs)
            _print_eval_results(eval_results, "Phoenix")

        else:
            comparison = EvaluationComparison()
            _comparison_instance = comparison
            
            total = len(agent_outputs)
            for i, tc in enumerate(agent_outputs, 1):
                console.print(f"  [{i}/{total}] Evaluating [cyan]{tc['test_case_id']}[/cyan]...")
                result = await comparison.evaluate_with_both(
                    test_case_id=tc["test_case_id"],
                    input_query=tc["input_query"],
                    actual_output=tc["actual_output"],
                    context=tc.get("context"),
                    expected_output=tc.get("expected_output"),
                    expected_tools=tc.get("expected_tools"),
                    actual_tools_called=tc.get("actual_tools_called"),
                    api_results=tc.get("api_results"),
                )
                _partial_results.append(result)
            
            comparison.print_comparison_report(_partial_results)
            filepath = comparison.save_results(_partial_results)
            console.print(f"\n[dim]Results saved to: {filepath}[/dim]")

    asyncio.run(run())


@cli.command()
def pipeline() -> None:
    """Run full pipeline: generate inputs, run agent, evaluate, log."""
    global _partial_results, _comparison_instance, _agent_outputs
    
    console.print(Panel("Full Pipeline Execution", title="Pipeline"))

    async def run() -> None:
        global _partial_results, _comparison_instance, _agent_outputs
        
        agent = AgentPlanner()
        comparison = EvaluationComparison()
        _comparison_instance = comparison
        _partial_results = []
        _agent_outputs = []

        console.print("\n[bold]Step 1: Generating test inputs[/bold]")
        console.print(f"  Using {len(TEST_CASES)} predefined test cases")

        console.print("\n[bold]Step 2: Running agent on test cases[/bold]")
        agent_outputs: list[dict[str, Any]] = []

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for i, tc in enumerate(TEST_CASES):
                    task = progress.add_task(
                        f"[{i+1}/{len(TEST_CASES)}] {tc['test_case_id']}...",
                        total=None,
                    )
                    try:
                        response = await agent.run(tc["input_query"])
                        actual_tools = [tool_call.tool_name for tool_call in response.tool_calls]
                        api_results = [
                            {
                                "tool_name": call.tool_name,
                                "parameters": call.parameters,
                                "result": call.result,
                                "success": call.success,
                                "error_message": call.error_message,
                                "latency_ms": call.latency_ms,
                            }
                            for call in response.tool_calls
                        ]
                        output_entry = {
                            "test_case_id": tc["test_case_id"],
                            "input_query": tc["input_query"],
                            "actual_output": response.report.model_dump_json(),
                            "expected_output": tc.get("expected_output"),
                            "expected_tools": tc.get("expected_tools"),
                            "actual_tools_called": actual_tools,
                            "api_results": api_results,
                        }
                        agent_outputs.append(output_entry)
                        _agent_outputs.append(output_entry)
                    except Exception as e:
                        console.print(f"  [red]Error on {tc['test_case_id']}: {e}[/red]")
                        error_entry = {
                            "test_case_id": tc["test_case_id"],
                            "input_query": tc["input_query"],
                            "actual_output": f"Error: {e}",
                            "actual_tools_called": [],
                            "api_results": [],
                        }
                        agent_outputs.append(error_entry)
                        _agent_outputs.append(error_entry)
                    progress.update(task, completed=True)

        finally:
            await agent.close()

        console.print("\n[bold]Step 3: Evaluating with both frameworks[/bold]")
        total = len(agent_outputs)
        for i, tc in enumerate(agent_outputs, 1):
            console.print(f"  [{i}/{total}] Evaluating [cyan]{tc['test_case_id']}[/cyan]...")
            result = await comparison.evaluate_with_both(
                test_case_id=tc["test_case_id"],
                input_query=tc["input_query"],
                actual_output=tc["actual_output"],
                context=tc.get("context"),
                expected_output=tc.get("expected_output"),
                expected_tools=tc.get("expected_tools"),
                actual_tools_called=tc.get("actual_tools_called"),
                api_results=tc.get("api_results"),
            )
            _partial_results.append(result)
            console.print(f"    [green]✓[/green] Complete ({len(result.phoenix_scores)} Phoenix, {len(result.deepeval_scores)} DeepEval metrics)")

        console.print("\n[bold]Step 4: Generating comparison report[/bold]")
        comparison.print_comparison_report(_partial_results)

        console.print("\n[bold]Step 5: Logging results[/bold]")
        filepath = comparison.save_results(_partial_results)
        console.print(f"  Results saved to: {filepath}")

        summary = _generate_pipeline_summary(_partial_results)
        _print_pipeline_summary(summary)

    asyncio.run(run())


@cli.command()
def report() -> None:
    """View comparison report from last evaluation run."""
    logs_dir = Path("logs")
    log_files = sorted(logs_dir.glob("evaluation_*.json"), reverse=True)

    if not log_files:
        console.print("[yellow]No evaluation logs found.[/yellow]")
        return

    latest_file = log_files[0]
    console.print(f"[dim]Loading: {latest_file}[/dim]\n")

    with open(latest_file) as f:
        data = json.load(f)

    console.print(Panel("Last Evaluation Report", title="Report"))

    summary = data.get("summary", {})
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Total Test Cases", str(summary.get("total_test_cases", 0)))
    table.add_row("Passed", f"[green]{summary.get('passed', 0)}[/green]")
    table.add_row("Failed", f"[red]{summary.get('failed', 0)}[/red]")
    table.add_row("Pass Rate", f"{summary.get('pass_rate', 0):.0%}")
    table.add_row("Avg DeepEval Score", f"{summary.get('avg_deepeval_score', 0):.2f}")
    table.add_row("Avg Phoenix Score", f"{summary.get('avg_phoenix_score', 0):.2f}")

    console.print(table)


@cli.command("drift-check")
def drift_check() -> None:
    """Analyze evaluation scores over time for drift detection."""
    console.print(Panel("Drift Detection Analysis", title="Drift Check"))

    logs_dir = Path("logs")
    log_files = sorted(logs_dir.glob("evaluation_*.json"))

    if len(log_files) < 2:
        console.print("[yellow]Need at least 2 evaluation runs for drift detection.[/yellow]")
        return

    console.print(f"[dim]Analyzing {len(log_files)} evaluation runs...[/dim]\n")

    scores_over_time: list[dict[str, Any]] = []
    for filepath in log_files:
        with open(filepath) as f:
            data = json.load(f)
            scores_over_time.append({
                "timestamp": data.get("timestamp"),
                "summary": data.get("summary", {}),
            })

    table = Table(show_header=True, header_style="bold")
    table.add_column("Timestamp", style="dim")
    table.add_column("Pass Rate", justify="right")
    table.add_column("DeepEval Avg", justify="right")
    table.add_column("Phoenix Avg", justify="right")
    table.add_column("Status", justify="center")

    baseline = scores_over_time[0]["summary"]
    threshold = 0.1

    for item in scores_over_time:
        summary = item["summary"]
        ts = item["timestamp"][:19] if item["timestamp"] else "N/A"

        pass_rate = summary.get("pass_rate", 0)
        de_avg = summary.get("avg_deepeval_score", 0)
        ph_avg = summary.get("avg_phoenix_score", 0)

        baseline_pass = baseline.get("pass_rate", 0)
        drift = pass_rate < (baseline_pass - threshold)

        status = "[red]DRIFT[/red]" if drift else "[green]OK[/green]"

        table.add_row(
            ts,
            f"{pass_rate:.0%}",
            f"{de_avg:.2f}",
            f"{ph_avg:.2f}",
            status,
        )

    console.print(table)

    latest = scores_over_time[-1]["summary"]
    console.print("\n[bold]Drift Analysis:[/bold]")
    console.print(
        f"  Baseline pass rate: {baseline.get('pass_rate', 0):.0%}"
    )
    console.print(
        f"  Current pass rate: {latest.get('pass_rate', 0):.0%}"
    )

    change = latest.get("pass_rate", 0) - baseline.get("pass_rate", 0)
    if change < -threshold:
        console.print(
            f"\n[red]⚠ DRIFT DETECTED: Pass rate dropped by {abs(change):.0%}[/red]"
        )
    else:
        console.print("\n[green]✓ No significant drift detected[/green]")


def _print_eval_results(results: list[Any], framework: str) -> None:
    """Print evaluation results in a formatted table."""
    table = Table(show_header=True, header_style="bold cyan", title=f"{framework} Results")
    table.add_column("Test Case", style="dim")
    table.add_column("Passed", justify="center")
    table.add_column("Scores", justify="left")

    for result in results:
        scores_str = ", ".join(
            f"{s.metric_name.split('_')[-1]}: {s.score:.2f}"
            for s in (result.deepeval_scores or result.phoenix_scores)[:3]
        )
        passed = "[green]✓[/green]" if result.overall_passed else "[red]✗[/red]"
        table.add_row(result.test_case_id, passed, scores_str)

    console.print(table)


def _generate_pipeline_summary(results: list[Any]) -> dict[str, Any]:
    """Generate summary statistics from pipeline results."""
    total = len(results)
    passed = sum(1 for r in results if r.overall_passed)

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0,
    }


def _print_pipeline_summary(summary: dict[str, Any]) -> None:
    """Print pipeline summary."""
    console.print("\n" + "=" * 60)
    console.print("[bold]PIPELINE SUMMARY[/bold]")
    console.print("=" * 60)
    console.print(f"  Total test cases: {summary['total']}")
    console.print(f"  Passed: [green]{summary['passed']}[/green]")
    console.print(f"  Failed: [red]{summary['failed']}[/red]")
    console.print(f"  Pass rate: {summary['pass_rate']:.0%}")
    console.print("=" * 60)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
