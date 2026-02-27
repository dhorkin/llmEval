"""Schema validation tests for output structures."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.schemas import (
    BookAnalysisReport,
    BookInfo,
    BookSearchResult,
    MealInfo,
    MealRecommendation,
    MealRecommendationReport,
    NEOInfo,
    NEOReport,
    NutrientInfo,
    PoemInfo,
    PoetryAnalysisReport,
    PoetryResult,
)


class TestBookSchemas:
    """Test book-related schemas."""

    def test_book_info_valid(self) -> None:
        """Test: Valid BookInfo schema."""
        book = BookInfo(
            title="1984",
            author="George Orwell",
            year=1949,
            subjects=["dystopia", "politics"],
        )
        assert book.title == "1984"
        assert book.year == 1949

    def test_book_info_optional_fields(self) -> None:
        """Test: BookInfo with optional fields."""
        book = BookInfo(title="Test", author="Test Author")
        assert book.year is None
        assert book.subjects == []

    def test_book_search_result_valid(self) -> None:
        """Test: Valid BookSearchResult schema."""
        result = BookSearchResult(
            query_author="Orwell",
            books=[BookInfo(title="1984", author="Orwell")],
            total_found=1,
        )
        assert result.total_found == 1
        assert not result.no_results

    def test_book_analysis_report_valid(self) -> None:
        """Test: Valid BookAnalysisReport schema."""
        report = BookAnalysisReport(
            query="Find books by Orwell",
            books_found=[],
            theme_summary="No books found",
            common_themes=[],
        )
        assert report.type == "BOOK_ANALYSIS"


class TestNEOSchemas:
    """Test NASA NEO schemas."""

    def test_neo_info_valid(self) -> None:
        """Test: Valid NEOInfo schema."""
        neo = NEOInfo(
            name="2024 AB",
            neo_id="12345",
            close_approach_date="2024-03-01",
            is_potentially_hazardous=True,
        )
        assert neo.is_potentially_hazardous

    def test_neo_report_risk_levels(self) -> None:
        """Test: NEOReport risk level validation."""
        report = NEOReport(
            query="NEO check",
            date_range="2024-03-01 to 2024-03-07",
            objects_count=5,
            hazardous_count=1,
            risk_level="elevated",
            risk_assessment="Elevated risk detected",
        )
        assert report.risk_level == "elevated"

        report_normal = NEOReport(
            query="NEO check",
            date_range="2024-03-01 to 2024-03-07",
            objects_count=5,
            hazardous_count=0,
            risk_level="normal",
            risk_assessment="Normal activity",
        )
        assert report_normal.risk_level == "normal"

    def test_neo_report_invalid_risk_level(self) -> None:
        """Test: NEOReport rejects invalid risk level."""
        with pytest.raises(ValidationError):
            NEOReport(
                query="NEO check",
                date_range="2024-03-01 to 2024-03-07",
                risk_level="critical",  # type: ignore[arg-type]
                risk_assessment="Invalid",
            )


class TestPoetrySchemas:
    """Test poetry-related schemas."""

    def test_poem_info_valid(self) -> None:
        """Test: Valid PoemInfo schema."""
        poem = PoemInfo(
            title="Sonnet 18",
            author="William Shakespeare",
            lines=["Shall I compare thee to a summer's day?"],
            linecount=14,
        )
        assert poem.linecount == 14

    def test_poetry_result_no_results(self) -> None:
        """Test: PoetryResult with no results."""
        result = PoetryResult(
            query_author="Unknown",
            poems=[],
            total_found=0,
            no_results=True,
        )
        assert result.no_results

    def test_poetry_analysis_report_with_focus(self) -> None:
        """Test: PoetryAnalysisReport with focus lines."""
        poem = PoemInfo(
            title="Sonnet 18",
            author="Shakespeare",
            lines=["Line 1", "Line 2", "Line 3", "Line 4"],
            linecount=14,
        )
        report = PoetryAnalysisReport(
            query="Analyze quatrain",
            poem=poem,
            focus_lines=["Line 1", "Line 2", "Line 3", "Line 4"],
            analysis="Quatrain analysis",
            literary_devices=["metaphor"],
        )
        assert len(report.focus_lines) == 4


class TestNutritionSchemas:
    """Test nutrition-related schemas."""

    def test_nutrient_info_valid(self) -> None:
        """Test: Valid NutrientInfo schema."""
        nutrients = NutrientInfo(
            calories=450,
            protein_g=35,
            carbs_g=30,
            fat_g=20,
        )
        assert nutrients.calories == 450

    def test_meal_info_valid(self) -> None:
        """Test: Valid MealInfo schema."""
        meal = MealInfo(
            name="Grilled Salmon",
            description="Fresh salmon with vegetables",
            ingredients=["salmon", "olive oil", "vegetables"],
            diet_compatible=True,
            restriction_safe=True,
        )
        assert meal.restriction_safe

    def test_meal_recommendation_conflict(self) -> None:
        """Test: MealRecommendation with conflict."""
        result = MealRecommendation(
            diet_type="vegan",
            restrictions=[],
            meals=[],
            conflict_detected=True,
            conflict_message="Cannot combine vegan with beef",
        )
        assert result.conflict_detected

    def test_meal_recommendation_report_valid(self) -> None:
        """Test: Valid MealRecommendationReport schema."""
        report = MealRecommendationReport(
            query="Mediterranean meals",
            diet_type="mediterranean",
            restrictions=["dairy"],
            recommendations=[],
            nutritional_summary="No meals found",
        )
        assert report.type == "MEAL_RECOMMENDATION"


class TestReportTypeValidation:
    """Test report type literal values."""

    def test_book_report_type(self) -> None:
        """Test: BookAnalysisReport type is BOOK_ANALYSIS."""
        report = BookAnalysisReport(
            query="test",
            books_found=[],
            theme_summary="test",
        )
        assert report.type == "BOOK_ANALYSIS"

    def test_neo_report_type(self) -> None:
        """Test: NEOReport type is NEO_REPORT."""
        report = NEOReport(
            query="test",
            date_range="test",
            risk_assessment="test",
        )
        assert report.type == "NEO_REPORT"

    def test_poetry_report_type(self) -> None:
        """Test: PoetryAnalysisReport type is POETRY_ANALYSIS."""
        report = PoetryAnalysisReport(
            query="test",
            analysis="test",
        )
        assert report.type == "POETRY_ANALYSIS"

    def test_meal_report_type(self) -> None:
        """Test: MealRecommendationReport type is MEAL_RECOMMENDATION."""
        report = MealRecommendationReport(
            query="test",
            diet_type="balanced",
            nutritional_summary="test",
        )
        assert report.type == "MEAL_RECOMMENDATION"
