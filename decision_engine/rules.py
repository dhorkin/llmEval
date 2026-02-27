"""Decision engine with deterministic rules for each domain."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Literal

from models.schemas import (
    BookAnalysisReport,
    BookInfo,
    BookSearchResult,
    MealInfo,
    MealRecommendation,
    MealRecommendationReport,
    NEOFeedResult,
    NEOInfo,
    NEOReport,
    PoemInfo,
    PoetryAnalysisReport,
    PoetryResult,
)


class DecisionEngine:
    """Deterministic decision engine for processing tool outputs."""

    # ==========================================================================
    # Book Rules
    # ==========================================================================

    @staticmethod
    def process_book_results(
        query: str,
        search_result: BookSearchResult,
        theme_summary: str = "",
        common_themes: list[str] | None = None,
    ) -> BookAnalysisReport:
        """
        Apply book decision rules.

        Rules:
        - IF query contains year constraint: filter books by publish_date
        - IF no books found: return empty result with "no_results" flag
        - IF multiple books found: sort by relevance to query
        """
        books = search_result.books
        common_themes = common_themes or []

        year_before = DecisionEngine._extract_year_before(query)
        year_after = DecisionEngine._extract_year_after(query)

        if year_before is not None:
            books = [b for b in books if b.year is not None and b.year < year_before]

        if year_after is not None:
            books = [b for b in books if b.year is not None and b.year > year_after]

        if books:
            books = DecisionEngine._sort_books_by_relevance(books, query)

        no_results = len(books) == 0

        if no_results and not theme_summary:
            theme_summary = "No books found matching the criteria."

        return BookAnalysisReport(
            type="BOOK_ANALYSIS",
            query=query,
            books_found=books,
            theme_summary=theme_summary,
            common_themes=common_themes,
            no_results=no_results,
        )

    @staticmethod
    def _extract_year_before(query: str) -> int | None:
        """Extract 'before YYYY' year constraint from query."""
        patterns = [
            r"before\s+(\d{4})",
            r"prior\s+to\s+(\d{4})",
            r"pre-(\d{4})",
            r"earlier\s+than\s+(\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _extract_year_after(query: str) -> int | None:
        """Extract 'after YYYY' year constraint from query."""
        patterns = [
            r"after\s+(\d{4})",
            r"since\s+(\d{4})",
            r"post-(\d{4})",
            r"later\s+than\s+(\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _sort_books_by_relevance(books: list[BookInfo], query: str) -> list[BookInfo]:
        """Sort books by relevance to query (books with year first, then alphabetically)."""

        def relevance_score(book: BookInfo) -> tuple[int, int, str]:
            has_year = 0 if book.year else 1
            year = -(book.year or 0)
            title = book.title.lower()
            return (has_year, year, title)

        return sorted(books, key=relevance_score)

    # ==========================================================================
    # NASA NEO Rules
    # ==========================================================================

    @staticmethod
    def process_neo_results(
        query: str,
        feed_result: NEOFeedResult,
    ) -> NEOReport:
        """
        Apply NASA NEO decision rules.

        Rules:
        - IF query mentions "this week": date_range = current_week()
        - IF query mentions "this weekend": date_range = upcoming_weekend()
        - IF hazardous_count > 0: set risk_level = "elevated"
        - ELSE: set risk_level = "normal"
        """
        neos = feed_result.near_earth_objects
        hazardous_neos = [n for n in neos if n.is_potentially_hazardous]
        hazardous_count = len(hazardous_neos)

        if hazardous_count > 0:
            risk_level: Literal["normal", "elevated"] = "elevated"
            risk_assessment = (
                f"ELEVATED RISK: {hazardous_count} potentially hazardous asteroid(s) "
                f"detected in the date range. Monitor closely."
            )
        else:
            risk_level = "normal"
            risk_assessment = (
                "Normal activity levels. No potentially hazardous asteroids detected "
                "in the specified date range."
            )

        closest_approach: NEOInfo | None = None
        if neos:
            sorted_by_distance = sorted(
                neos, key=lambda x: x.miss_distance_km or float("inf")
            )
            closest_approach = sorted_by_distance[0] if sorted_by_distance else None

        date_range = f"{feed_result.start_date} to {feed_result.end_date}"

        return NEOReport(
            type="NEO_REPORT",
            query=query,
            date_range=date_range,
            objects_count=len(neos),
            hazardous_count=hazardous_count,
            closest_approach=closest_approach,
            risk_level=risk_level,
            risk_assessment=risk_assessment,
            near_earth_objects=neos,
        )

    @staticmethod
    def parse_date_intent(query: str) -> tuple[str, str] | None:
        """
        Parse date intent from query.

        Returns (start_date, end_date) tuple or None if no date pattern found.
        """
        query_lower = query.lower()
        today = datetime.now()

        if "this week" in query_lower:
            monday = today - timedelta(days=today.weekday())
            sunday = monday + timedelta(days=6)
            return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")

        if "this weekend" in query_lower or "weekend" in query_lower:
            days_until_saturday = (5 - today.weekday()) % 7
            if days_until_saturday == 0 and today.weekday() != 5:
                days_until_saturday = 7
            saturday = today + timedelta(days=days_until_saturday)
            sunday = saturday + timedelta(days=1)
            return saturday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")

        if "today" in query_lower:
            date_str = today.strftime("%Y-%m-%d")
            return date_str, date_str

        if "tomorrow" in query_lower:
            tomorrow = today + timedelta(days=1)
            date_str = tomorrow.strftime("%Y-%m-%d")
            return date_str, date_str

        if "next week" in query_lower:
            next_monday = today + timedelta(days=(7 - today.weekday()))
            next_sunday = next_monday + timedelta(days=6)
            return next_monday.strftime("%Y-%m-%d"), next_sunday.strftime("%Y-%m-%d")

        return None

    @staticmethod
    def validate_date(date_str: str) -> tuple[bool, str | None]:
        """
        Validate a date string.

        Returns (is_valid, error_message).
        """
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")

            if date.month == 2 and date.day > 29:
                return False, f"Invalid date: {date_str} (February has max 29 days)"
            if date.month in [4, 6, 9, 11] and date.day > 30:
                return False, f"Invalid date: {date_str} (month has max 30 days)"

            return True, None
        except ValueError as e:
            return False, f"Invalid date format: {date_str}. Error: {e}"

    # ==========================================================================
    # Poetry Rules
    # ==========================================================================

    @staticmethod
    def process_poetry_results(
        query: str,
        poetry_result: PoetryResult,
        analysis: str = "",
        literary_devices: list[str] | None = None,
    ) -> PoetryAnalysisReport:
        """
        Apply poetry decision rules.

        Rules:
        - IF query mentions "sonnet": filter for linecount == 14
        - IF query mentions "quatrain": focus_lines = lines[0:4]
        - IF query mentions "metaphor" or "imagery": include literary_analysis = True
        """
        literary_devices = literary_devices or []
        poems = poetry_result.poems

        if DecisionEngine._mentions_sonnet(query):
            poems = [p for p in poems if p.linecount == 14]

        selected_poem: PoemInfo | None = poems[0] if poems else None
        focus_lines: list[str] = []

        if selected_poem and DecisionEngine._mentions_quatrain(query):
            focus_lines = selected_poem.lines[:4]

        needs_literary_analysis = DecisionEngine._needs_literary_analysis(query)
        if needs_literary_analysis and not literary_devices:
            literary_devices = ["Analysis requested but not yet performed"]

        no_results = selected_poem is None

        if no_results and not analysis:
            analysis = "No poems found matching the criteria."

        return PoetryAnalysisReport(
            type="POETRY_ANALYSIS",
            query=query,
            poem=selected_poem,
            focus_lines=focus_lines,
            analysis=analysis,
            literary_devices=literary_devices,
            no_results=no_results,
        )

    @staticmethod
    def _mentions_sonnet(query: str) -> bool:
        """Check if query mentions sonnet."""
        return "sonnet" in query.lower()

    @staticmethod
    def _mentions_quatrain(query: str) -> bool:
        """Check if query mentions quatrain or first four lines."""
        query_lower = query.lower()
        return "quatrain" in query_lower or "first four" in query_lower or "first 4" in query_lower

    @staticmethod
    def _needs_literary_analysis(query: str) -> bool:
        """Check if query requires literary analysis."""
        analysis_terms = [
            "metaphor",
            "imagery",
            "symbolism",
            "analyze",
            "analysis",
            "explain",
            "meaning",
            "interpret",
            "theme",
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in analysis_terms)

    # ==========================================================================
    # Nutrition Rules
    # ==========================================================================

    @staticmethod
    def process_nutrition_results(
        query: str,
        meal_result: MealRecommendation,
    ) -> MealRecommendationReport:
        """
        Apply nutrition decision rules.

        Rules:
        - IF restrictions contains "dairy": exclude all dairy ingredients
        - IF restrictions contains "nuts": exclude all nut ingredients
        - IF diet_type not recognized: return error with supported_diets list
        """

        if meal_result.conflict_detected:
            return MealRecommendationReport(
                type="MEAL_RECOMMENDATION",
                query=query,
                diet_type=meal_result.diet_type,
                restrictions=meal_result.restrictions,
                recommendations=[],
                nutritional_summary="Unable to provide recommendations due to conflicting requirements.",
                conflict_detected=True,
                conflict_message=meal_result.conflict_message,
            )

        meals = meal_result.meals
        nutritional_summary = DecisionEngine._generate_nutritional_summary(meals)

        return MealRecommendationReport(
            type="MEAL_RECOMMENDATION",
            query=query,
            diet_type=meal_result.diet_type,
            restrictions=meal_result.restrictions,
            recommendations=meals,
            nutritional_summary=nutritional_summary,
            conflict_detected=False,
        )

    @staticmethod
    def _generate_nutritional_summary(meals: list[MealInfo]) -> str:
        """Generate a summary of nutritional information for meals."""
        if not meals:
            return "No meals available for nutritional summary."

        total_calories = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        count = 0

        for meal in meals:
            if meal.nutrients:
                if meal.nutrients.calories:
                    total_calories += meal.nutrients.calories
                if meal.nutrients.protein_g:
                    total_protein += meal.nutrients.protein_g
                if meal.nutrients.carbs_g:
                    total_carbs += meal.nutrients.carbs_g
                if meal.nutrients.fat_g:
                    total_fat += meal.nutrients.fat_g
                count += 1

        if count == 0:
            return "Nutritional information not available for these meals."

        avg_calories = total_calories / count
        avg_protein = total_protein / count
        avg_carbs = total_carbs / count
        avg_fat = total_fat / count

        return (
            f"Average per meal: {avg_calories:.0f} calories, "
            f"{avg_protein:.1f}g protein, {avg_carbs:.1f}g carbs, {avg_fat:.1f}g fat. "
            f"All {len(meals)} recommended meals comply with specified dietary restrictions."
        )

    @staticmethod
    def extract_diet_and_restrictions(query: str) -> tuple[str | None, list[str]]:
        """Extract diet type and restrictions from query."""
        from tools.nutrition_tool import COMMON_RESTRICTIONS, SUPPORTED_DIETS

        query_lower = query.lower()
        diet_type: str | None = None
        restrictions: list[str] = []

        for diet in SUPPORTED_DIETS:
            if diet in query_lower:
                diet_type = diet
                break

        for restriction in COMMON_RESTRICTIONS:
            avoid_patterns = [
                f"avoid {restriction}",
                f"no {restriction}",
                f"without {restriction}",
                f"{restriction}-free",
                f"exclude {restriction}",
            ]
            if any(p in query_lower for p in avoid_patterns):
                restrictions.append(restriction)

        return diet_type, restrictions

    # ==========================================================================
    # Intent Detection
    # ==========================================================================

    @staticmethod
    def detect_intent(query: str) -> str:
        """Detect the primary intent from a user query."""
        query_lower = query.lower()

        book_indicators = ["book", "author", "written by", "published", "novel", "read"]
        neo_indicators = ["neo", "asteroid", "near earth", "space", "nasa", "comet"]
        poetry_indicators = ["poem", "poetry", "sonnet", "verse", "poet", "quatrain"]
        nutrition_indicators = ["meal", "diet", "food", "eat", "recipe", "nutrition", "dinner", "lunch", "breakfast"]

        scores: dict[str, int] = {
            "book_analysis": sum(1 for i in book_indicators if i in query_lower),
            "neo_report": sum(1 for i in neo_indicators if i in query_lower),
            "poetry_analysis": sum(1 for i in poetry_indicators if i in query_lower),
            "meal_recommendation": sum(1 for i in nutrition_indicators if i in query_lower),
        }

        max_intent = max(scores, key=lambda k: scores[k])
        return max_intent if scores[max_intent] > 0 else "unknown"

    @staticmethod
    def determine_tools_needed(intent: str) -> list[str]:
        """Determine which tools are needed based on intent."""
        tool_mapping = {
            "book_analysis": ["book_search"],
            "neo_report": ["nasa_neo"],
            "poetry_analysis": ["poetry_search"],
            "meal_recommendation": ["nutrition_meal_recommendation"],
        }
        return tool_mapping.get(intent, [])
