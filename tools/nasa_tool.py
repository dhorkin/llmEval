"""NASA tool wrapper for NASA Open APIs (NEO - Near Earth Objects)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from models.schemas import NEOFeedResult, NEOInfo
from tools.base import BaseTool, ToolError


class NASATool(BaseTool):
    """Tool for querying NASA Near Earth Object (NEO) data."""

    BASE_URL = "https://api.nasa.gov/neo/rest/v1"

    @property
    def name(self) -> str:
        return "nasa_neo"

    @property
    def description(self) -> str:
        return (
            "Query NASA's Near Earth Object (NEO) database to find asteroids and comets "
            "passing near Earth. Can search by date range to find close approaches. "
            "Returns information about NEO name, size, velocity, miss distance, and "
            "whether it is potentially hazardous."
        )

    def _get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (max 7 days from start)",
                },
            },
            "required": ["start_date", "end_date"],
        }

    def _get_api_key(self) -> str:
        """Get NASA API key, falling back to DEMO_KEY."""
        return self.settings.nasa_api_key or "DEMO_KEY"

    async def execute(
        self,
        start_date: str,
        end_date: str,
    ) -> NEOFeedResult:
        """
        Get NEO feed for a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (max 7 days from start)

        Returns:
            NEOFeedResult with all NEOs in the date range
        """
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")
        self._validate_date_range(start_date, end_date)

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "api_key": self._get_api_key(),
        }

        try:
            response = await self.get(f"{self.BASE_URL}/feed", params=params)
            data = response.json()
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to parse response: {e}", self.name) from e

        neos: list[NEOInfo] = []
        neo_data = data.get("near_earth_objects", {})

        for date_str, date_neos in neo_data.items():
            for neo in date_neos:
                neo_info = self._parse_neo(neo, date_str)
                neos.append(neo_info)

        neos.sort(key=lambda x: x.miss_distance_km or float("inf"))

        return NEOFeedResult(
            start_date=start_date,
            end_date=end_date,
            element_count=data.get("element_count", len(neos)),
            near_earth_objects=neos,
        )

    def _validate_date(self, date_str: str, field_name: str) -> None:
        """Validate date format."""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            raise ToolError(
                f"Invalid {field_name} format. Expected YYYY-MM-DD, got: {date_str}",
                self.name,
                recoverable=False,
            ) from e

    def _validate_date_range(self, start_date: str, end_date: str) -> None:
        """Validate date range is within 7 days."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if end < start:
            raise ToolError(
                f"end_date ({end_date}) must be after start_date ({start_date})",
                self.name,
                recoverable=False,
            )

        if (end - start).days > 7:
            raise ToolError(
                "Date range cannot exceed 7 days",
                self.name,
                recoverable=False,
            )

    def _parse_neo(self, neo: dict[str, Any], date_str: str) -> NEOInfo:
        """Parse NEO data from API response."""
        close_approach = neo.get("close_approach_data", [{}])[0]
        diameter = neo.get("estimated_diameter", {}).get("kilometers", {})
        miss_distance = close_approach.get("miss_distance", {})
        relative_velocity = close_approach.get("relative_velocity", {})

        miss_km_str = miss_distance.get("kilometers")
        velocity_kph_str = relative_velocity.get("kilometers_per_hour")

        return NEOInfo(
            name=neo.get("name", "Unknown"),
            neo_id=neo.get("id", ""),
            absolute_magnitude=neo.get("absolute_magnitude_h"),
            estimated_diameter_min_km=diameter.get("estimated_diameter_min"),
            estimated_diameter_max_km=diameter.get("estimated_diameter_max"),
            is_potentially_hazardous=neo.get("is_potentially_hazardous_asteroid", False),
            close_approach_date=date_str,
            relative_velocity_kph=float(velocity_kph_str) if velocity_kph_str else None,
            miss_distance_km=float(miss_km_str) if miss_km_str else None,
            orbiting_body=close_approach.get("orbiting_body", "Earth"),
        )

    @staticmethod
    def get_current_week_range() -> tuple[str, str]:
        """Get date range for current week (Monday to Sunday)."""
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")

    @staticmethod
    def get_upcoming_weekend_range() -> tuple[str, str]:
        """Get date range for upcoming weekend (Saturday to Sunday)."""
        today = datetime.now()
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0 and today.weekday() != 5:
            days_until_saturday = 7
        saturday = today + timedelta(days=days_until_saturday)
        sunday = saturday + timedelta(days=1)
        return saturday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")
