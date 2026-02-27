"""Regression tests for book-related decision engine rules."""

from __future__ import annotations

from syrupy.assertion import SnapshotAssertion

from conftest import make_book_info
from decision_engine.rules import DecisionEngine
from models.schemas import BookSearchResult


class TestBookRegressionEdgeCases:
    """Regression tests for book edge cases."""

    def test_unknown_author_returns_empty_results(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Edge Case (edge_001): Unknown author should return no results."""
        search_result = BookSearchResult(
            query_author="Unknown Author XYZ",
            books=[],
            total_found=0,
            no_results=True,
        )

        report = DecisionEngine.process_book_results(
            query="Find books by Unknown Author XYZ",
            search_result=search_result,
        )

        assert report.no_results is True
        assert report.books_found == []
        assert snapshot == report.model_dump()

    def test_temporal_impossibility_einstein_after_2020(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Failure Case (failure_001): Einstein books after 2020 is impossible."""
        search_result = BookSearchResult(
            query_author="Albert Einstein",
            books=[
                make_book_info("Relativity: The Special and General Theory", "Albert Einstein", 1916),
                make_book_info("The Meaning of Relativity", "Albert Einstein", 1922),
                make_book_info("Ideas and Opinions", "Albert Einstein", 1954),
            ],
            total_found=3,
        )

        report = DecisionEngine.process_book_results(
            query="Find books by Einstein published after 2020",
            search_result=search_result,
        )

        assert report.no_results is True
        assert report.books_found == []
        assert snapshot == report.model_dump()

    def test_year_filter_before_constraint(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Year filter with 'before' constraint works correctly."""
        search_result = BookSearchResult(
            query_author="George Orwell",
            books=[
                make_book_info("Animal Farm", "George Orwell", 1945),
                make_book_info("1984", "George Orwell", 1949),
                make_book_info("Coming Up for Air", "George Orwell", 1939),
            ],
            total_found=3,
        )

        report = DecisionEngine.process_book_results(
            query="Find books by George Orwell published before 1946",
            search_result=search_result,
        )

        assert len(report.books_found) == 2
        assert all(b.year < 1946 for b in report.books_found if b.year)
        assert snapshot == report.model_dump()

    def test_year_filter_after_constraint(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Year filter with 'after' constraint works correctly."""
        search_result = BookSearchResult(
            query_author="George Orwell",
            books=[
                make_book_info("Animal Farm", "George Orwell", 1945),
                make_book_info("1984", "George Orwell", 1949),
                make_book_info("Coming Up for Air", "George Orwell", 1939),
            ],
            total_found=3,
        )

        report = DecisionEngine.process_book_results(
            query="Find books by George Orwell published after 1940",
            search_result=search_result,
        )

        assert len(report.books_found) == 2
        assert all(b.year > 1940 for b in report.books_found if b.year)
        assert snapshot == report.model_dump()

    def test_books_sorted_by_relevance(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Books are sorted by relevance (year desc, then title)."""
        search_result = BookSearchResult(
            query_author="Test Author",
            books=[
                make_book_info("Z Book", "Test Author", 2000),
                make_book_info("A Book", "Test Author", 2010),
                make_book_info("M Book", "Test Author", 2005),
                make_book_info("No Year Book", "Test Author", None),
            ],
            total_found=4,
        )

        report = DecisionEngine.process_book_results(
            query="Find books by Test Author",
            search_result=search_result,
        )

        assert len(report.books_found) == 4
        assert report.books_found[0].year == 2010
        assert snapshot == report.model_dump()
