"""Regression tests for nutrition/meal decision engine rules."""

from __future__ import annotations

import pytest
from syrupy.assertion import SnapshotAssertion

from decision_engine.rules import DecisionEngine
from models.schemas import MealInfo, MealRecommendation, NutrientInfo


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
                MealInfo(
                    name="Grilled Salmon with Quinoa",
                    description="Fresh salmon fillet with herb quinoa",
                    ingredients=["salmon", "quinoa", "olive oil", "lemon"],
                    nutrients=NutrientInfo(
                        calories=450.0,
                        protein_g=35.0,
                        carbs_g=30.0,
                        fat_g=20.0,
                    ),
                    diet_compatible=True,
                    restriction_safe=True,
                ),
                MealInfo(
                    name="Mediterranean Vegetable Bake",
                    description="Roasted vegetables with herbs",
                    ingredients=["zucchini", "eggplant", "tomatoes", "olive oil"],
                    nutrients=NutrientInfo(
                        calories=280.0,
                        protein_g=8.0,
                        carbs_g=35.0,
                        fat_g=14.0,
                    ),
                    diet_compatible=True,
                    restriction_safe=True,
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
                MealInfo(
                    name="Chicken Stir Fry",
                    description="Quick chicken stir fry",
                    ingredients=["chicken", "vegetables", "rice"],
                    nutrients=NutrientInfo(
                        calories=400.0,
                        protein_g=30.0,
                        carbs_g=40.0,
                        fat_g=12.0,
                    ),
                ),
                MealInfo(
                    name="Pasta Primavera",
                    description="Vegetable pasta dish",
                    ingredients=["pasta", "vegetables", "olive oil"],
                    nutrients=NutrientInfo(
                        calories=350.0,
                        protein_g=12.0,
                        carbs_g=55.0,
                        fat_g=10.0,
                    ),
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
