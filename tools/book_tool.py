"""Book tool wrapper for Open Library API."""

from __future__ import annotations

from typing import Any

from models.schemas import BookInfo, BookSearchResult
from tools.base import BaseTool, ToolError


class BookTool(BaseTool):
    """Tool for searching books via Open Library API."""

    BASE_URL = "https://openlibrary.org"

    @property
    def name(self) -> str:
        return "book_search"

    @property
    def description(self) -> str:
        return (
            "Search for books by author, title, or subject using Open Library. "
            "Can filter by publication year. Returns book information including "
            "title, author, year, subjects, and description."
        )

    def _get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "author": {
                    "type": "string",
                    "description": "Author name to search for",
                },
                "title": {
                    "type": "string",
                    "description": "Book title to search for (optional)",
                },
                "published_before": {
                    "type": "integer",
                    "description": "Filter books published before this year (optional)",
                },
                "published_after": {
                    "type": "integer",
                    "description": "Filter books published after this year (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10)",
                    "default": 10,
                },
            },
            "required": ["author"],
        }

    async def execute(
        self,
        author: str,
        title: str | None = None,
        published_before: int | None = None,
        published_after: int | None = None,
        limit: int = 10,
    ) -> BookSearchResult:
        """
        Search for books by author with optional filters.

        Args:
            author: Author name to search for
            title: Optional title filter
            published_before: Filter books published before this year
            published_after: Filter books published after this year
            limit: Maximum number of results

        Returns:
            BookSearchResult with matching books
        """
        query_parts = [f"author:{author}"]
        if title:
            query_parts.append(f"title:{title}")

        query = " ".join(query_parts)
        params = {
            "q": query,
            "limit": limit,
            "fields": "key,title,author_name,first_publish_year,subject,isbn",
        }

        try:
            response = await self.get(f"{self.BASE_URL}/search.json", params=params)
            data = response.json()
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to parse response: {e}", self.name) from e

        books: list[BookInfo] = []
        docs = data.get("docs", [])

        for doc in docs:
            year = doc.get("first_publish_year")

            if published_before is not None and year is not None:
                if year >= published_before:
                    continue

            if published_after is not None and year is not None:
                if year <= published_after:
                    continue

            authors = doc.get("author_name", [])
            author_str = authors[0] if authors else "Unknown"

            subjects = doc.get("subject", [])[:10]

            isbns = doc.get("isbn", [])
            isbn = isbns[0] if isbns else None

            book = BookInfo(
                title=doc.get("title", "Unknown Title"),
                author=author_str,
                year=year,
                subjects=subjects,
                description=None,
                isbn=isbn,
                open_library_key=doc.get("key"),
            )
            books.append(book)

        return BookSearchResult(
            query_author=author,
            query_title=title,
            books=books,
            total_found=len(books),
            no_results=len(books) == 0,
        )

    async def get_book_description(self, work_key: str) -> str | None:
        """Fetch description for a specific work."""
        try:
            response = await self.get(f"{self.BASE_URL}{work_key}.json")
            data = response.json()
            description = data.get("description")
            if isinstance(description, dict):
                return description.get("value")
            return description
        except Exception:
            return None
