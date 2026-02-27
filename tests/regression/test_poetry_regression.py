"""Regression tests for poetry decision engine rules."""

from __future__ import annotations

from syrupy.assertion import SnapshotAssertion

from conftest import make_poem_info
from decision_engine.rules import DecisionEngine
from models.schemas import PoetryResult


class TestPoetryRegressionEdgeCases:
    """Regression tests for poetry edge cases."""

    def test_sonnet_filter_applied(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Sonnet filter selects only 14-line poems."""
        poetry_result = PoetryResult(
            query_author="William Shakespeare",
            poems=[
                make_poem_info("Sonnet 18", "William Shakespeare", ["Shall I compare thee to a summer's day?"] * 14),
                make_poem_info("Short Poem", "William Shakespeare", ["A short poem line"] * 8, linecount=8),
                make_poem_info("Sonnet 29", "William Shakespeare", ["When, in disgrace with fortune"] * 14),
            ],
            total_found=3,
        )

        report = DecisionEngine.process_poetry_results(
            query="Find a sonnet by Shakespeare",
            poetry_result=poetry_result,
        )

        assert report.poem is not None
        assert report.poem.linecount == 14
        assert snapshot == report.model_dump()

    def test_quatrain_focus_lines(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Quatrain request extracts first 4 lines."""
        quatrain_lines = [
            "Shall I compare thee to a summer's day?",
            "Thou art more lovely and more temperate:",
            "Rough winds do shake the darling buds of May,",
            "And summer's lease hath all too short a date:",
            "Sometime too hot the eye of heaven shines,",
            "And often is his gold complexion dimm'd;",
        ]
        poetry_result = PoetryResult(
            query_author="William Shakespeare",
            poems=[make_poem_info("Sonnet 18", "William Shakespeare", quatrain_lines, linecount=14)],
            total_found=1,
        )

        report = DecisionEngine.process_poetry_results(
            query="Analyze the first quatrain of a Shakespeare sonnet",
            poetry_result=poetry_result,
        )

        assert len(report.focus_lines) == 4
        assert report.focus_lines[0] == "Shall I compare thee to a summer's day?"
        assert snapshot == report.model_dump()

    def test_no_poems_found(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Edge Case: No poems matching criteria."""
        poetry_result = PoetryResult(
            query_author="Nonexistent Poet",
            poems=[],
            total_found=0,
            no_results=True,
        )

        report = DecisionEngine.process_poetry_results(
            query="Find poems by Nonexistent Poet",
            poetry_result=poetry_result,
        )

        assert report.no_results is True
        assert report.poem is None
        assert "No poems found" in report.analysis
        assert snapshot == report.model_dump()

    def test_literary_analysis_requested(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Literary analysis flag is set for metaphor/imagery queries."""
        poetry_result = PoetryResult(
            query_author="Robert Frost",
            poems=[make_poem_info("The Road Not Taken", "Robert Frost", ["Two roads diverged in a yellow wood,"] * 20, linecount=20)],
            total_found=1,
        )

        report = DecisionEngine.process_poetry_results(
            query="Analyze the metaphor in The Road Not Taken",
            poetry_result=poetry_result,
        )

        assert len(report.literary_devices) > 0
        assert snapshot == report.model_dump()

    def test_intent_detection_poetry(self) -> None:
        """Regression: Poetry intent is detected correctly."""
        intent = DecisionEngine.detect_intent(
            "Find a poem about love by Emily Dickinson"
        )

        assert intent == "poetry_analysis"
