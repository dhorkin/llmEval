"""Functional tests for tool usage verification."""

from __future__ import annotations

from decision_engine.rules import DecisionEngine
from tools.book_tool import BookTool
from tools.nasa_tool import NASATool
from tools.nutrition_tool import NutritionTool
from tools.poetry_tool import PoetryTool


class TestDecisionEngineIntentDetection:
    """Test intent detection functionality."""

    def test_book_intent_detection(self, sample_book_query: str) -> None:
        """Test: Detect book intent from query."""
        intent = DecisionEngine.detect_intent(sample_book_query)
        assert intent == "book_analysis"

    def test_neo_intent_detection(self, sample_neo_query: str) -> None:
        """Test: Detect NEO intent from query."""
        intent = DecisionEngine.detect_intent(sample_neo_query)
        assert intent == "neo_report"

    def test_poetry_intent_detection(self, sample_poetry_query: str) -> None:
        """Test: Detect poetry intent from query."""
        intent = DecisionEngine.detect_intent(sample_poetry_query)
        assert intent == "poetry_analysis"

    def test_nutrition_intent_detection(self, sample_nutrition_query: str) -> None:
        """Test: Detect nutrition intent from query."""
        intent = DecisionEngine.detect_intent(sample_nutrition_query)
        assert intent == "meal_recommendation"


class TestDecisionEngineToolDetermination:
    """Test tool determination functionality."""

    def test_book_tools_needed(self) -> None:
        """Test: Determine book search tool needed."""
        tools = DecisionEngine.determine_tools_needed("book_analysis")
        assert "book_search" in tools

    def test_neo_tools_needed(self) -> None:
        """Test: Determine NASA NEO tool needed."""
        tools = DecisionEngine.determine_tools_needed("neo_report")
        assert "nasa_neo" in tools

    def test_poetry_tools_needed(self) -> None:
        """Test: Determine poetry search tool needed."""
        tools = DecisionEngine.determine_tools_needed("poetry_analysis")
        assert "poetry_search" in tools

    def test_nutrition_tools_needed(self) -> None:
        """Test: Determine nutrition tool needed."""
        tools = DecisionEngine.determine_tools_needed("meal_recommendation")
        assert "nutrition_meal_recommendation" in tools


class TestDateParsing:
    """Test date parsing functionality."""

    def test_this_week_parsing(self) -> None:
        """Test: Parse 'this week' date intent."""
        result = DecisionEngine.parse_date_intent("Check NEO data for this week")
        assert result is not None
        start_date, end_date = result
        assert len(start_date) == 10
        assert len(end_date) == 10

    def test_this_weekend_parsing(self) -> None:
        """Test: Parse 'this weekend' date intent."""
        result = DecisionEngine.parse_date_intent("What asteroids pass by this weekend?")
        assert result is not None
        start_date, end_date = result
        assert len(start_date) == 10
        assert len(end_date) == 10

    def test_today_parsing(self) -> None:
        """Test: Parse 'today' date intent."""
        result = DecisionEngine.parse_date_intent("NEO report for today")
        assert result is not None
        start_date, end_date = result
        assert start_date == end_date


class TestYearExtraction:
    """Test year extraction from queries."""

    def test_before_year_extraction(self) -> None:
        """Test: Extract 'before YYYY' constraint."""
        year = DecisionEngine._extract_year_before("Books published before 1950")
        assert year == 1950

    def test_after_year_extraction(self) -> None:
        """Test: Extract 'after YYYY' constraint."""
        year = DecisionEngine._extract_year_after("Books published after 2000")
        assert year == 2000

    def test_no_year_constraint(self) -> None:
        """Test: No year constraint returns None."""
        year = DecisionEngine._extract_year_before("Find books by Orwell")
        assert year is None


class TestPoetryRules:
    """Test poetry-specific rules."""

    def test_sonnet_detection(self) -> None:
        """Test: Detect sonnet mention in query."""
        assert DecisionEngine._mentions_sonnet("Find a sonnet by Shakespeare")
        assert not DecisionEngine._mentions_sonnet("Find a poem by Shakespeare")

    def test_quatrain_detection(self) -> None:
        """Test: Detect quatrain mention in query."""
        assert DecisionEngine._mentions_quatrain("Analyze the first quatrain")
        assert DecisionEngine._mentions_quatrain("Explain the first four lines")
        assert not DecisionEngine._mentions_quatrain("Analyze the poem")

    def test_literary_analysis_needed(self) -> None:
        """Test: Detect literary analysis requirement."""
        assert DecisionEngine._needs_literary_analysis("Explain the metaphor")
        assert DecisionEngine._needs_literary_analysis("Analyze the imagery")
        assert not DecisionEngine._needs_literary_analysis("Find a poem")


class TestNutritionRules:
    """Test nutrition-specific rules."""

    def test_diet_extraction(self) -> None:
        """Test: Extract diet type from query."""
        diet, restrictions = DecisionEngine.extract_diet_and_restrictions(
            "Mediterranean diet meals"
        )
        assert diet == "mediterranean"

    def test_restriction_extraction(self) -> None:
        """Test: Extract dietary restrictions from query."""
        diet, restrictions = DecisionEngine.extract_diet_and_restrictions(
            "Vegan meals without nuts and avoid dairy"
        )
        assert "nuts" in restrictions
        assert "dairy" in restrictions


class TestToolSchemas:
    """Test tool schema generation."""

    def test_book_tool_schema(self) -> None:
        """Test: BookTool generates valid schema."""
        tool = BookTool()
        schema = tool.get_schema()
        assert schema["name"] == "book_search"
        assert "parameters" in schema
        assert "author" in schema["parameters"]["properties"]

    def test_nasa_tool_schema(self) -> None:
        """Test: NASATool generates valid schema."""
        tool = NASATool()
        schema = tool.get_schema()
        assert schema["name"] == "nasa_neo"
        assert "start_date" in schema["parameters"]["properties"]

    def test_poetry_tool_schema(self) -> None:
        """Test: PoetryTool generates valid schema."""
        tool = PoetryTool()
        schema = tool.get_schema()
        assert schema["name"] == "poetry_search"
        assert "author" in schema["parameters"]["properties"]

    def test_nutrition_tool_schema(self) -> None:
        """Test: NutritionTool generates valid schema."""
        tool = NutritionTool()
        schema = tool.get_schema()
        assert schema["name"] == "nutrition_meal_recommendation"
        assert "diet_type" in schema["parameters"]["properties"]
