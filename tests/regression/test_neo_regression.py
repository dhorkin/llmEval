"""Regression tests for NASA NEO decision engine rules."""

from __future__ import annotations

from syrupy.assertion import SnapshotAssertion

from decision_engine.rules import DecisionEngine
from models.schemas import NEOFeedResult, NEOInfo


class TestNEORegressionEdgeCases:
    """Regression tests for NEO edge cases."""

    def test_invalid_date_february_30(self) -> None:
        """Edge Case: February 30th is an invalid date."""
        is_valid, error_message = DecisionEngine.validate_date("2024-02-30")

        assert is_valid is False
        assert error_message is not None
        assert "Invalid date" in error_message or "February" in error_message

    def test_invalid_date_format(self) -> None:
        """Edge Case: Invalid date format is rejected."""
        is_valid, error_message = DecisionEngine.validate_date("30-02-2024")

        assert is_valid is False
        assert error_message is not None
        assert "Invalid date format" in error_message

    def test_valid_date_accepted(self) -> None:
        """Regression: Valid dates are accepted."""
        is_valid, error_message = DecisionEngine.validate_date("2024-03-15")

        assert is_valid is True
        assert error_message is None

    def test_leap_year_february_29_valid(self) -> None:
        """Regression: February 29 in leap year is valid."""
        is_valid, error_message = DecisionEngine.validate_date("2024-02-29")

        assert is_valid is True
        assert error_message is None

    def test_hazardous_asteroids_elevate_risk(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Hazardous asteroids set risk level to elevated."""
        feed_result = NEOFeedResult(
            start_date="2024-03-01",
            end_date="2024-03-02",
            element_count=3,
            near_earth_objects=[
                NEOInfo(
                    name="2024 AB",
                    neo_id="12345",
                    is_potentially_hazardous=True,
                    close_approach_date="2024-03-01",
                    miss_distance_km=2500000.0,
                ),
                NEOInfo(
                    name="2024 CD",
                    neo_id="67890",
                    is_potentially_hazardous=False,
                    close_approach_date="2024-03-01",
                    miss_distance_km=5000000.0,
                ),
                NEOInfo(
                    name="2024 EF",
                    neo_id="11111",
                    is_potentially_hazardous=True,
                    close_approach_date="2024-03-02",
                    miss_distance_km=1000000.0,
                ),
            ],
        )

        report = DecisionEngine.process_neo_results(
            query="Check for hazardous asteroids this week",
            feed_result=feed_result,
        )

        assert report.risk_level == "elevated"
        assert report.hazardous_count == 2
        assert "ELEVATED RISK" in report.risk_assessment
        assert snapshot == report.model_dump()

    def test_no_hazardous_asteroids_normal_risk(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: No hazardous asteroids means normal risk level."""
        feed_result = NEOFeedResult(
            start_date="2024-03-01",
            end_date="2024-03-02",
            element_count=2,
            near_earth_objects=[
                NEOInfo(
                    name="2024 GH",
                    neo_id="22222",
                    is_potentially_hazardous=False,
                    close_approach_date="2024-03-01",
                    miss_distance_km=8000000.0,
                ),
                NEOInfo(
                    name="2024 IJ",
                    neo_id="33333",
                    is_potentially_hazardous=False,
                    close_approach_date="2024-03-02",
                    miss_distance_km=10000000.0,
                ),
            ],
        )

        report = DecisionEngine.process_neo_results(
            query="Check for asteroids near Earth",
            feed_result=feed_result,
        )

        assert report.risk_level == "normal"
        assert report.hazardous_count == 0
        assert "Normal activity" in report.risk_assessment
        assert snapshot == report.model_dump()

    def test_closest_approach_identified(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Closest approach asteroid is correctly identified."""
        feed_result = NEOFeedResult(
            start_date="2024-03-01",
            end_date="2024-03-02",
            element_count=3,
            near_earth_objects=[
                NEOInfo(
                    name="Far Away",
                    neo_id="44444",
                    is_potentially_hazardous=False,
                    close_approach_date="2024-03-01",
                    miss_distance_km=10000000.0,
                ),
                NEOInfo(
                    name="Close One",
                    neo_id="55555",
                    is_potentially_hazardous=False,
                    close_approach_date="2024-03-01",
                    miss_distance_km=500000.0,
                ),
                NEOInfo(
                    name="Medium Distance",
                    neo_id="66666",
                    is_potentially_hazardous=False,
                    close_approach_date="2024-03-02",
                    miss_distance_km=3000000.0,
                ),
            ],
        )

        report = DecisionEngine.process_neo_results(
            query="What is the closest asteroid approach?",
            feed_result=feed_result,
        )

        assert report.closest_approach is not None
        assert report.closest_approach.name == "Close One"
        assert report.closest_approach.miss_distance_km == 500000.0
        assert snapshot == report.model_dump()

    def test_date_intent_parsing_this_week(self) -> None:
        """Regression: 'this week' intent is parsed correctly."""
        result = DecisionEngine.parse_date_intent("Show NEOs for this week")

        assert result is not None
        start_date, end_date = result
        assert start_date is not None
        assert end_date is not None

    def test_date_intent_parsing_today(self) -> None:
        """Regression: 'today' intent is parsed correctly."""
        result = DecisionEngine.parse_date_intent("Show NEOs for today")

        assert result is not None
        start_date, end_date = result
        assert start_date == end_date
