"""Poetry tool wrapper for PoetryDB API."""

from __future__ import annotations

from typing import Any

from models.schemas import PoemInfo, PoetryResult
from tools.base import BaseTool, ToolError


class PoetryTool(BaseTool):
    """Tool for searching poems via PoetryDB API."""

    BASE_URL = "https://poetrydb.org"

    @property
    def name(self) -> str:
        return "poetry_search"

    @property
    def description(self) -> str:
        return (
            "Search for poems by author or title using PoetryDB. "
            "Can filter by form (sonnet, haiku, etc.) based on line count. "
            "Returns poem title, author, full text lines, and line count."
        )

    def _get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "author": {
                    "type": "string",
                    "description": "Poet name to search for",
                },
                "title": {
                    "type": "string",
                    "description": "Poem title to search for (optional)",
                },
                "linecount": {
                    "type": "integer",
                    "description": "Filter by exact line count (e.g., 14 for sonnets)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10)",
                    "default": 10,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        author: str | None = None,
        title: str | None = None,
        linecount: int | None = None,
        limit: int = 10,
    ) -> PoetryResult:
        """
        Search for poems by author and/or title.

        Args:
            author: Poet name to search for
            title: Poem title to search for
            linecount: Filter by exact line count (e.g., 14 for sonnets)
            limit: Maximum number of results

        Returns:
            PoetryResult with matching poems
        """
        if not author and not title:
            raise ToolError(
                "At least one of author or title must be provided",
                self.name,
                recoverable=False,
            )

        url = self._build_search_url(author, title, linecount)

        try:
            response = await self.get(url)
            data = response.json()
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to parse response: {e}", self.name) from e

        if isinstance(data, dict) and data.get("status") == 404:
            return PoetryResult(
                query_author=author,
                query_title=title,
                poems=[],
                total_found=0,
                no_results=True,
            )

        poems: list[PoemInfo] = []

        if isinstance(data, list):
            for poem_data in data[:limit]:
                poem = self._parse_poem(poem_data)
                if linecount is None or poem.linecount == linecount:
                    poems.append(poem)

        return PoetryResult(
            query_author=author,
            query_title=title,
            poems=poems,
            total_found=len(poems),
            no_results=len(poems) == 0,
        )

    def _build_search_url(
        self,
        author: str | None,
        title: str | None,
        linecount: int | None,
    ) -> str:
        """Build the PoetryDB search URL."""
        search_fields: list[str] = []
        search_values: list[str] = []

        if author:
            search_fields.append("author")
            search_values.append(author)

        if title:
            search_fields.append("title")
            search_values.append(title)

        if linecount:
            search_fields.append("linecount")
            search_values.append(str(linecount))

        fields_str = ",".join(search_fields)
        values_str = ";".join(search_values)

        return f"{self.BASE_URL}/{fields_str}/{values_str}"

    def _parse_poem(self, poem_data: dict[str, Any]) -> PoemInfo:
        """Parse poem data from API response."""
        lines = poem_data.get("lines", [])
        return PoemInfo(
            title=poem_data.get("title", "Unknown"),
            author=poem_data.get("author", "Unknown"),
            lines=lines,
            linecount=poem_data.get("linecount", len(lines)),
        )

    async def get_sonnets_by_author(self, author: str, limit: int = 10) -> PoetryResult:
        """Convenience method to get sonnets (14-line poems) by an author."""
        return await self.execute(author=author, linecount=14, limit=limit)

    @staticmethod
    def get_quatrain(poem: PoemInfo) -> list[str]:
        """Extract the first quatrain (4 lines) from a poem."""
        return poem.lines[:4] if len(poem.lines) >= 4 else poem.lines

    @staticmethod
    def get_form_by_linecount(linecount: int) -> str | None:
        """Determine poem form based on line count."""
        form_mapping = {
            14: "sonnet",
            3: "haiku",
            5: "limerick",
            19: "villanelle",
        }
        return form_mapping.get(linecount)

    @staticmethod
    def is_sonnet(poem: PoemInfo) -> bool:
        """Check if a poem is a sonnet (14 lines)."""
        return poem.linecount == 14
