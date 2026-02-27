"""Regression tests for nutrition/meal decision engine rules."""

from __future__ import annotations

from syrupy.assertion import SnapshotAssertion

from conftest import make_meal_info
from decision_engine.rules import DecisionEngine
from models.schemas import MealRecommendation


class TestNutritionRegressionEdgeCases:
    """Regression tests for nutrition edge cases."""

    def test_conflicting_restrictions_vegan_with_beef(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Edge Case: Vegan diet with beef is a conflict."""
        meal_result = MealRecommendation(
            diet_type="vegan",
            restrictions=[],
            meals=[],
            conflict_detected=True,
            conflict_message="Cannot recommend vegan meals containing beef. Vegan diet excludes all animal products.",
        )

        report = DecisionEngine.process_nutrition_results(
            query="Recommend a vegan meal with beef",
            meal_result=meal_result,
        )

        assert report.conflict_detected is True
        assert report.recommendations == []
        assert report.conflict_message is not None
        assert "vegan" in report.conflict_message.lower()
        assert snapshot == report.model_dump()

    def test_valid_diet_with_restrictions(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Valid diet type with restrictions works correctly."""
        meal_result = MealRecommendation(
            diet_type="mediterranean",
            restrictions=["dairy", "nuts"],
            meals=[
                make_meal_info(
                    "Grilled Salmon with Quinoa",
                    "Fresh salmon fillet with herb quinoa",
                    ["salmon", "quinoa", "olive oil", "lemon"],
                    calories=450, protein=35, carbs=30, fat=20,
                ),
                make_meal_info(
                    "Mediterranean Vegetable Bake",
                    "Roasted vegetables with herbs",
                    ["zucchini", "eggplant", "tomatoes", "olive oil"],
                    calories=280, protein=8, carbs=35, fat=14,
                ),
            ],
            conflict_detected=False,
        )

        report = DecisionEngine.process_nutrition_results(
            query="Mediterranean dinner options without dairy and nuts",
            meal_result=meal_result,
        )

        assert report.conflict_detected is False
        assert len(report.recommendations) == 2
        assert all(m.restriction_safe for m in report.recommendations)
        assert snapshot == report.model_dump()

    def test_nutritional_summary_generated(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Nutritional summary is generated correctly."""
        meal_result = MealRecommendation(
            diet_type="balanced",
            restrictions=[],
            meals=[
                make_meal_info(
                    "Chicken Stir Fry",
                    "Quick chicken stir fry",
                    ["chicken", "vegetables", "rice"],
                    calories=400, protein=30, carbs=40, fat=12,
                ),
                make_meal_info(
                    "Pasta Primavera",
                    "Vegetable pasta dish",
                    ["pasta", "vegetables", "olive oil"],
                    calories=350, protein=12, carbs=55, fat=10,
                ),
            ],
            conflict_detected=False,
        )

        report = DecisionEngine.process_nutrition_results(
            query="Recommend balanced dinner options",
            meal_result=meal_result,
        )

        assert report.nutritional_summary is not None
        assert "calories" in report.nutritional_summary.lower()
        assert "protein" in report.nutritional_summary.lower()
        assert snapshot == report.model_dump()

    def test_empty_meals_summary(
        self, snapshot: SnapshotAssertion
    ) -> None:
        """Regression: Empty meal list produces appropriate summary."""
        meal_result = MealRecommendation(
            diet_type="keto",
            restrictions=["all-common-foods"],
            meals=[],
            conflict_detected=False,
        )

        report = DecisionEngine.process_nutrition_results(
            query="Keto meals avoiding all common foods",
            meal_result=meal_result,
        )

        assert report.recommendations == []
        assert "not available" in report.nutritional_summary.lower() or "no meals" in report.nutritional_summary.lower()
        assert snapshot == report.model_dump()

    def test_diet_and_restrictions_extraction(self) -> None:
        """Regression: Diet type and restrictions are extracted from query."""
        diet, restrictions = DecisionEngine.extract_diet_and_restrictions(
            "I want a vegan meal without nuts and no dairy please"
        )

        assert diet == "vegan"
        assert "nuts" in restrictions
        assert "dairy" in restrictions

    def test_intent_detection_nutrition(self) -> None:
        """Regression: Nutrition intent is detected correctly."""
        intent = DecisionEngine.detect_intent(
            "What should I eat for dinner tonight?"
        )

        assert intent == "meal_recommendation"

    def test_intent_detection_multiple_signals(self) -> None:
        """Regression: Intent with multiple signals is resolved correctly."""
        intent = DecisionEngine.detect_intent(
            "Find me healthy meal recipes for a Mediterranean diet"
        )

        assert intent == "meal_recommendation"
