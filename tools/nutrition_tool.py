"""Nutrition tool wrapper for LogMeal Food AI API."""

from __future__ import annotations

from typing import Any

from models.schemas import MealInfo, MealRecommendation, NutrientInfo
from tools.base import BaseTool, ToolError


SUPPORTED_DIETS = [
    "mediterranean",
    "vegetarian",
    "vegan",
    "keto",
    "paleo",
    "low-carb",
    "gluten-free",
    "balanced",
]

COMMON_RESTRICTIONS = [
    "dairy",
    "nuts",
    "gluten",
    "eggs",
    "soy",
    "shellfish",
    "fish",
    "pork",
    "beef",
    "chicken",
]

RESTRICTION_INGREDIENTS: dict[str, list[str]] = {
    "dairy": ["milk", "cheese", "cream", "butter", "yogurt", "whey"],
    "nuts": ["almond", "walnut", "pecan", "cashew", "peanut", "hazelnut", "pistachio"],
    "gluten": ["wheat", "barley", "rye", "bread", "pasta", "flour"],
    "eggs": ["egg", "eggs", "mayonnaise"],
    "soy": ["soy", "tofu", "tempeh", "edamame", "soybean"],
    "shellfish": ["shrimp", "crab", "lobster", "clam", "mussel", "oyster"],
    "fish": ["salmon", "tuna", "cod", "tilapia", "fish"],
    "pork": ["pork", "bacon", "ham", "sausage"],
    "beef": ["beef", "steak", "ground beef", "veal"],
    "chicken": ["chicken", "poultry"],
}

MEAL_DATABASE: dict[str, list[dict[str, Any]]] = {
    "mediterranean": [
        {
            "name": "Grilled Salmon with Quinoa",
            "description": "Fresh Atlantic salmon grilled with lemon and herbs, served with quinoa and roasted vegetables",
            "ingredients": ["salmon", "quinoa", "olive oil", "lemon", "garlic", "zucchini", "bell pepper"],
            "calories": 450,
            "protein_g": 35,
            "carbs_g": 30,
            "fat_g": 22,
            "fiber_g": 5,
        },
        {
            "name": "Greek Salad with Grilled Chicken",
            "description": "Classic Greek salad topped with grilled chicken breast and feta cheese",
            "ingredients": ["chicken", "cucumber", "tomato", "feta cheese", "olive oil", "olives", "red onion"],
            "calories": 380,
            "protein_g": 32,
            "carbs_g": 15,
            "fat_g": 24,
            "fiber_g": 4,
        },
        {
            "name": "Lemon Herb Chicken with Roasted Vegetables",
            "description": "Tender chicken thighs marinated in lemon and Mediterranean herbs with seasonal vegetables",
            "ingredients": ["chicken", "lemon", "oregano", "garlic", "eggplant", "tomato", "olive oil"],
            "calories": 420,
            "protein_g": 38,
            "carbs_g": 18,
            "fat_g": 20,
            "fiber_g": 6,
        },
        {
            "name": "Hummus and Vegetable Platter",
            "description": "Creamy hummus served with fresh vegetables and whole grain pita",
            "ingredients": ["chickpeas", "tahini", "olive oil", "carrot", "cucumber", "bell pepper", "pita"],
            "calories": 320,
            "protein_g": 12,
            "carbs_g": 40,
            "fat_g": 14,
            "fiber_g": 10,
        },
        {
            "name": "Mediterranean Baked Fish",
            "description": "White fish baked with tomatoes, capers, and olives in a light wine sauce",
            "ingredients": ["cod", "tomato", "capers", "olives", "white wine", "olive oil", "parsley"],
            "calories": 280,
            "protein_g": 30,
            "carbs_g": 10,
            "fat_g": 12,
            "fiber_g": 3,
        },
    ],
    "vegetarian": [
        {
            "name": "Vegetable Stir-Fry with Tofu",
            "description": "Crispy tofu with mixed vegetables in a savory sauce over brown rice",
            "ingredients": ["tofu", "broccoli", "carrot", "soy sauce", "ginger", "garlic", "brown rice"],
            "calories": 380,
            "protein_g": 18,
            "carbs_g": 45,
            "fat_g": 15,
            "fiber_g": 8,
        },
        {
            "name": "Caprese Pasta",
            "description": "Fresh pasta with tomatoes, mozzarella, and basil in olive oil",
            "ingredients": ["pasta", "tomato", "mozzarella", "basil", "olive oil", "garlic"],
            "calories": 450,
            "protein_g": 16,
            "carbs_g": 55,
            "fat_g": 18,
            "fiber_g": 4,
        },
        {
            "name": "Vegetable Curry",
            "description": "Rich coconut curry with mixed vegetables and chickpeas",
            "ingredients": ["chickpeas", "coconut milk", "spinach", "potato", "curry spices", "rice"],
            "calories": 420,
            "protein_g": 14,
            "carbs_g": 50,
            "fat_g": 18,
            "fiber_g": 10,
        },
    ],
    "vegan": [
        {
            "name": "Buddha Bowl",
            "description": "Colorful bowl with quinoa, roasted vegetables, and tahini dressing",
            "ingredients": ["quinoa", "chickpeas", "sweet potato", "kale", "tahini", "lemon"],
            "calories": 450,
            "protein_g": 16,
            "carbs_g": 60,
            "fat_g": 16,
            "fiber_g": 14,
        },
        {
            "name": "Lentil Soup",
            "description": "Hearty lentil soup with vegetables and Mediterranean spices",
            "ingredients": ["lentils", "carrot", "celery", "onion", "tomato", "cumin", "olive oil"],
            "calories": 320,
            "protein_g": 18,
            "carbs_g": 45,
            "fat_g": 8,
            "fiber_g": 16,
        },
        {
            "name": "Stuffed Bell Peppers",
            "description": "Bell peppers stuffed with rice, black beans, and vegetables",
            "ingredients": ["bell pepper", "rice", "black beans", "corn", "tomato", "cumin"],
            "calories": 350,
            "protein_g": 12,
            "carbs_g": 55,
            "fat_g": 8,
            "fiber_g": 12,
        },
    ],
    "keto": [
        {
            "name": "Grilled Steak with Asparagus",
            "description": "Juicy ribeye steak with grilled asparagus and herb butter",
            "ingredients": ["beef", "asparagus", "butter", "garlic", "rosemary"],
            "calories": 520,
            "protein_g": 45,
            "carbs_g": 6,
            "fat_g": 36,
            "fiber_g": 3,
        },
        {
            "name": "Salmon with Avocado Salsa",
            "description": "Pan-seared salmon topped with fresh avocado salsa",
            "ingredients": ["salmon", "avocado", "tomato", "lime", "cilantro", "olive oil"],
            "calories": 480,
            "protein_g": 38,
            "carbs_g": 8,
            "fat_g": 34,
            "fiber_g": 6,
        },
        {
            "name": "Chicken Caesar Salad (No Croutons)",
            "description": "Classic Caesar salad with grilled chicken, no croutons",
            "ingredients": ["chicken", "romaine", "parmesan", "caesar dressing", "lemon"],
            "calories": 420,
            "protein_g": 35,
            "carbs_g": 5,
            "fat_g": 30,
            "fiber_g": 2,
        },
    ],
    "balanced": [
        {
            "name": "Grilled Chicken with Sweet Potato",
            "description": "Seasoned grilled chicken breast with baked sweet potato and steamed broccoli",
            "ingredients": ["chicken", "sweet potato", "broccoli", "olive oil", "herbs"],
            "calories": 420,
            "protein_g": 35,
            "carbs_g": 35,
            "fat_g": 12,
            "fiber_g": 7,
        },
        {
            "name": "Turkey Meatballs with Zucchini Noodles",
            "description": "Lean turkey meatballs served over zucchini noodles with marinara",
            "ingredients": ["turkey", "zucchini", "tomato", "garlic", "basil", "olive oil"],
            "calories": 350,
            "protein_g": 28,
            "carbs_g": 20,
            "fat_g": 18,
            "fiber_g": 5,
        },
    ],
}


class NutritionTool(BaseTool):
    """Tool for meal recommendations based on diet and restrictions."""

    BASE_URL = "https://api.logmeal.es/v2"

    @property
    def name(self) -> str:
        return "nutrition_meal_recommendation"

    @property
    def description(self) -> str:
        return (
            "Get meal recommendations based on diet type and dietary restrictions. "
            f"Supported diets: {', '.join(SUPPORTED_DIETS)}. "
            f"Common restrictions: {', '.join(COMMON_RESTRICTIONS)}. "
            "Returns meal suggestions with nutritional information."
        )

    def _get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "diet_type": {
                    "type": "string",
                    "description": f"Diet type: {', '.join(SUPPORTED_DIETS)}",
                    "enum": SUPPORTED_DIETS,
                },
                "restrictions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Dietary restrictions: {', '.join(COMMON_RESTRICTIONS)}",
                },
                "meal_count": {
                    "type": "integer",
                    "description": "Number of meal recommendations (default 3)",
                    "default": 3,
                },
            },
            "required": ["diet_type"],
        }

    async def execute(
        self,
        diet_type: str,
        restrictions: list[str] | None = None,
        meal_count: int = 3,
    ) -> MealRecommendation:
        """
        Get meal recommendations based on diet and restrictions.

        Args:
            diet_type: Type of diet (mediterranean, vegan, etc.)
            restrictions: List of dietary restrictions
            meal_count: Number of meals to recommend

        Returns:
            MealRecommendation with filtered meals
        """
        restrictions = restrictions or []
        diet_lower = diet_type.lower()

        if diet_lower not in SUPPORTED_DIETS:
            raise ToolError(
                f"Unsupported diet type: {diet_type}. Supported: {', '.join(SUPPORTED_DIETS)}",
                self.name,
                recoverable=False,
            )

        conflict = self._check_diet_restriction_conflict(diet_lower, restrictions)
        if conflict:
            return MealRecommendation(
                diet_type=diet_type,
                restrictions=restrictions,
                meals=[],
                conflict_detected=True,
                conflict_message=conflict,
            )

        meals = self._get_meals_for_diet(diet_lower, restrictions, meal_count)

        return MealRecommendation(
            diet_type=diet_type,
            restrictions=restrictions,
            meals=meals,
            conflict_detected=False,
        )

    def _check_diet_restriction_conflict(
        self, diet_type: str, restrictions: list[str]
    ) -> str | None:
        """Check for conflicts between diet and restrictions."""
        conflicts = {
            ("vegan", "beef"): None,
            ("vegan", "chicken"): None,
            ("vegan", "fish"): None,
            ("vegan", "pork"): None,
            ("vegetarian", "beef"): None,
            ("vegetarian", "chicken"): None,
            ("vegetarian", "fish"): None,
            ("vegetarian", "pork"): None,
        }

        meat_in_vegan = any(r in ["beef", "chicken", "pork", "fish"] for r in restrictions)
        if diet_type == "vegan" and meat_in_vegan:
            return None

        if "beef" in restrictions and diet_type not in ["vegan", "vegetarian"]:
            pass

        if "beef" in restrictions or "chicken" in restrictions or "pork" in restrictions:
            if diet_type in ["keto", "paleo"]:
                all_meat_restricted = all(
                    m in restrictions for m in ["beef", "chicken", "pork", "fish"]
                )
                if all_meat_restricted:
                    return f"{diet_type.title()} diet requires protein sources, but all meat options are restricted."

        return None

    def _get_meals_for_diet(
        self, diet_type: str, restrictions: list[str], meal_count: int
    ) -> list[MealInfo]:
        """Get filtered meals for diet type respecting restrictions."""
        base_meals = MEAL_DATABASE.get(diet_type, MEAL_DATABASE["balanced"])
        filtered_meals: list[MealInfo] = []

        for meal_data in base_meals:
            if self._meal_passes_restrictions(meal_data, restrictions):
                meal = MealInfo(
                    name=meal_data["name"],
                    description=meal_data["description"],
                    ingredients=meal_data["ingredients"],
                    nutrients=NutrientInfo(
                        calories=meal_data.get("calories"),
                        protein_g=meal_data.get("protein_g"),
                        carbs_g=meal_data.get("carbs_g"),
                        fat_g=meal_data.get("fat_g"),
                        fiber_g=meal_data.get("fiber_g"),
                    ),
                    diet_compatible=True,
                    restriction_safe=True,
                )
                filtered_meals.append(meal)

            if len(filtered_meals) >= meal_count:
                break

        return filtered_meals

    def _meal_passes_restrictions(
        self, meal_data: dict[str, Any], restrictions: list[str]
    ) -> bool:
        """Check if a meal passes all dietary restrictions."""
        ingredients = [ing.lower() for ing in meal_data.get("ingredients", [])]

        for restriction in restrictions:
            restriction_lower = restriction.lower()
            forbidden = RESTRICTION_INGREDIENTS.get(restriction_lower, [restriction_lower])

            for ingredient in ingredients:
                for forbidden_item in forbidden:
                    if forbidden_item in ingredient:
                        return False

        return True

    @staticmethod
    def get_supported_diets() -> list[str]:
        """Get list of supported diet types."""
        return SUPPORTED_DIETS.copy()

    @staticmethod
    def get_common_restrictions() -> list[str]:
        """Get list of common dietary restrictions."""
        return COMMON_RESTRICTIONS.copy()
