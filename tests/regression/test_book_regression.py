"""Regression tests for book-related decision engine rules."""

from __future__ import annotations

import pytest
from syrupy.assertion import SnapshotAssertion

from decision_engine.rules import DecisionEngine
from models.schemas import BookInfo, BookSearchResult


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
                BookInfo(
                    title="Relativity: The Special and General Theory",
                    author="Albert Einstein",
                    year=1916,
                ),
                BookInfo(
                    title="The Meaning of Relativity",
                    author="Albert Einstein",
                    year=1922,
                ),
                BookInfo(
                    title="Ideas and Opinions",
                    author="Albert Einstein",
                    year=1954,
                ),
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
                BookInfo(title="Animal Farm", author="George Orwell", year=1945),
                BookInfo(title="1984", author="George Orwell", year=1949),
                BookInfo(title="Coming Up for Air", author="George Orwell", year=1939),
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
                BookInfo(title="Animal Farm", author="George Orwell", year=1945),
                BookInfo(title="1984", author="George Orwell", year=1949),
                BookInfo(title="Coming Up for Air", author="George Orwell", year=1939),
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
                BookInfo(title="Z Book", author="Test Author", year=2000),
                BookInfo(title="A Book", author="Test Author", year=2010),
                BookInfo(title="M Book", author="Test Author", year=2005),
                BookInfo(title="No Year Book", author="Test Author", year=None),
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
