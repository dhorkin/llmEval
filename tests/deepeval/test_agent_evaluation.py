"""DeepEval test cases for agent evaluation."""

from __future__ import annotations

from deepeval import assert_test  # type: ignore[attr-defined]
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evaluation.deepeval_runner import SchemaValidationMetric, ToolCorrectnessMetric


class TestBookQueries:
    """Test cases for book-related queries."""

    def test_book_search_with_date_filter(self) -> None:
        """Test: Book search with date filter (Open Library)."""
        test_case = LLMTestCase(
            input="Find all books written by George Orwell published before 1950",
            actual_output="""
            {
                "type": "BOOK_ANALYSIS",
                "query": "Find all books written by George Orwell published before 1950",
                "books_found": [
                    {"title": "Animal Farm", "author": "George Orwell", "year": 1945},
                    {"title": "Coming Up for Air", "author": "George Orwell", "year": 1939},
                    {"title": "Homage to Catalonia", "author": "George Orwell", "year": 1938}
                ],
                "theme_summary": "Orwell's pre-1950 works explore themes of totalitarianism, social injustice, and the working class experience.",
                "common_themes": ["totalitarianism", "social criticism", "political allegory"]
            }
            """,
            expected_output="A list of George Orwell books published before 1950 with theme analysis",
            retrieval_context=[
                "George Orwell wrote Animal Farm in 1945",
                "George Orwell wrote 1984 in 1949",
                "George Orwell wrote Coming Up for Air in 1939",
                "George Orwell wrote Homage to Catalonia in 1938",
            ],
        )

        relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        assert_test(test_case, [relevancy_metric])

    def test_book_search_no_results(self) -> None:
        """Regression: Book search returns no results for unknown author."""
        test_case = LLMTestCase(
            input="Find books by Unknown Author XYZ",
            actual_output="""
            {
                "type": "BOOK_ANALYSIS",
                "query": "Find books by Unknown Author XYZ",
                "books_found": [],
                "theme_summary": "No books found matching the criteria.",
                "common_themes": [],
                "no_results": true
            }
            """,
            expected_output="Empty result with no_results flag set to true",
        )

        schema_metric = SchemaValidationMetric(threshold=0.8)
        assert_test(test_case, [schema_metric])


class TestNASAQueries:
    """Test cases for NASA NEO queries."""

    def test_neo_date_range(self) -> None:
        """Test: NEO check for date range (NASA)."""
        test_case = LLMTestCase(
            input="Check if there are any Near Earth Objects passing by Earth this weekend",
            actual_output="""
            {
                "type": "NEO_REPORT",
                "query": "Check if there are any Near Earth Objects passing by Earth this weekend",
                "date_range": "2024-03-02 to 2024-03-03",
                "objects_count": 5,
                "hazardous_count": 1,
                "closest_approach": {
                    "name": "2024 AB",
                    "miss_distance_km": 2500000,
                    "is_potentially_hazardous": false
                },
                "risk_level": "elevated",
                "risk_assessment": "ELEVATED RISK: 1 potentially hazardous asteroid detected."
            }
            """,
            expected_output="NEO report for the weekend with risk assessment",
            retrieval_context=[
                "NASA NEO API returns Near Earth Objects data",
                "Potentially hazardous asteroids are tracked by NASA",
            ],
        )

        relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        tool_metric = ToolCorrectnessMetric(
            expected_tools=["nasa_neo"],
            actual_tools_called=["nasa_neo"],
        )
        assert_test(test_case, [relevancy_metric, tool_metric])

    def test_neo_hazardous_detection(self) -> None:
        """Regression: NEO report flags hazardous objects correctly."""
        test_case = LLMTestCase(
            input="Are there any potentially hazardous asteroids near Earth today?",
            actual_output="""
            {
                "type": "NEO_REPORT",
                "query": "Are there any potentially hazardous asteroids near Earth today?",
                "hazardous_count": 2,
                "risk_level": "elevated",
                "risk_assessment": "ELEVATED RISK: 2 potentially hazardous asteroids detected."
            }
            """,
            expected_output="NEO report indicating elevated risk with hazardous count > 0",
        )

        schema_metric = SchemaValidationMetric(threshold=0.8)
        assert_test(test_case, [schema_metric])


class TestPoetryQueries:
    """Test cases for poetry queries."""

    def test_poetry_sonnet_search(self) -> None:
        """Test: Poetry search with form requirement (PoetryDB)."""
        test_case = LLMTestCase(
            input="Find a sonnet by William Shakespeare and explain the metaphor used in the first quatrain",
            actual_output="""
            {
                "type": "POETRY_ANALYSIS",
                "query": "Find a sonnet by William Shakespeare and explain the metaphor used in the first quatrain",
                "poem": {
                    "title": "Sonnet 18",
                    "author": "William Shakespeare",
                    "linecount": 14
                },
                "focus_lines": [
                    "Shall I compare thee to a summer's day?",
                    "Thou art more lovely and more temperate:",
                    "Rough winds do shake the darling buds of May,",
                    "And summer's lease hath all too short a date:"
                ],
                "analysis": "Shakespeare uses an extended metaphor comparing the beloved to a summer's day, then subverts it by declaring the beloved superior to summer.",
                "literary_devices": ["metaphor", "personification", "rhetorical question"]
            }
            """,
            expected_output="A Shakespeare sonnet with analysis of the first quatrain's metaphor",
            retrieval_context=[
                "Shakespeare's Sonnet 18 begins 'Shall I compare thee to a summer's day?'",
                "Sonnets have 14 lines",
                "A quatrain is 4 lines",
            ],
        )

        relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        faithfulness_metric = FaithfulnessMetric(threshold=0.8)
        assert_test(test_case, [relevancy_metric, faithfulness_metric])

    def test_poetry_no_match(self) -> None:
        """Regression: Poetry search for non-existent form by author."""
        test_case = LLMTestCase(
            input="Find a limerick by Shakespeare",
            actual_output="""
            {
                "type": "POETRY_ANALYSIS",
                "query": "Find a limerick by Shakespeare",
                "poem": null,
                "analysis": "No poems found matching the criteria. Shakespeare did not write limericks.",
                "no_results": true
            }
            """,
            expected_output="No results found indicating form mismatch",
        )

        schema_metric = SchemaValidationMetric(threshold=0.8)
        assert_test(test_case, [schema_metric])


class TestNutritionQueries:
    """Test cases for nutrition/meal queries."""

    def test_meal_recommendation_with_restrictions(self) -> None:
        """Test: Meal recommendation with restrictions (LogMeal)."""
        test_case = LLMTestCase(
            input="Based on a Mediterranean diet, recommend three dinner options that avoid dairy and nuts",
            actual_output="""
            {
                "type": "MEAL_RECOMMENDATION",
                "query": "Based on a Mediterranean diet, recommend three dinner options that avoid dairy and nuts",
                "diet_type": "mediterranean",
                "restrictions": ["dairy", "nuts"],
                "recommendations": [
                    {"name": "Grilled Salmon with Quinoa", "restriction_safe": true},
                    {"name": "Lemon Herb Chicken", "restriction_safe": true},
                    {"name": "Mediterranean Baked Fish", "restriction_safe": true}
                ],
                "nutritional_summary": "Average per meal: 400 calories, 35g protein. All meals are dairy and nut free."
            }
            """,
            expected_output="Three Mediterranean dinner options without dairy or nuts",
            retrieval_context=[
                "Mediterranean diet focuses on fish, olive oil, vegetables",
                "Dairy-free means no milk, cheese, yogurt",
                "Nut-free means no almonds, walnuts, etc.",
            ],
        )

        relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        faithfulness_metric = FaithfulnessMetric(threshold=0.8)
        assert_test(test_case, [relevancy_metric, faithfulness_metric])

    def test_meal_conflicting_restrictions(self) -> None:
        """Regression: Meal recommendation with impossible restrictions."""
        test_case = LLMTestCase(
            input="Recommend a vegan meal with beef",
            actual_output="""
            {
                "type": "MEAL_RECOMMENDATION",
                "query": "Recommend a vegan meal with beef",
                "diet_type": "vegan",
                "restrictions": [],
                "recommendations": [],
                "conflict_detected": true,
                "conflict_message": "Cannot recommend vegan meals containing beef. Vegan diet excludes all animal products."
            }
            """,
            expected_output="Conflict error indicating vegan cannot include beef",
        )

        schema_metric = SchemaValidationMetric(threshold=0.8)
        assert_test(test_case, [schema_metric])


class TestEdgeCases:
    """Edge case and failure case tests."""

    def test_invalid_date_validation(self) -> None:
        """Edge Case: Invalid date handling (February 30th)."""
        test_case = LLMTestCase(
            input="Check NEO data for February 30th",
            actual_output="""
            {
                "type": "VALIDATION_ERROR",
                "error_code": "INVALID_DATE",
                "message": "Invalid date: February 30th does not exist",
                "suggestion": "February has at most 29 days in a leap year"
            }
            """,
            expected_output="Validation error for invalid date",
        )

        schema_metric = SchemaValidationMetric(threshold=0.5)
        assert_test(test_case, [schema_metric])

    def test_hallucination_detection(self) -> None:
        """Known Failure: Detects hallucinated books."""
        test_case = LLMTestCase(
            input="Find all books about quantum physics by Einstein published after 2020",
            actual_output="""
            {
                "type": "BOOK_ANALYSIS",
                "query": "Find all books about quantum physics by Einstein published after 2020",
                "books_found": [],
                "theme_summary": "No books found. Albert Einstein passed away in 1955 and could not have published books after 2020.",
                "no_results": true
            }
            """,
            expected_output="Empty result acknowledging Einstein died in 1955",
            retrieval_context=[
                "Albert Einstein died on April 18, 1955",
                "No new Einstein books can be published by him after 1955",
            ],
        )

        faithfulness_metric = FaithfulnessMetric(threshold=0.8)
        assert_test(test_case, [faithfulness_metric])
