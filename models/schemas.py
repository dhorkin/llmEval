"""Pydantic schemas for all tool outputs and agent responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Book Tool Schemas (Open Library API)
# =============================================================================


class BookInfo(BaseModel):
    """Information about a single book."""

    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    year: int | None = Field(default=None, description="Publication year")
    subjects: list[str] = Field(default_factory=list, description="Book subjects/genres")
    description: str | None = Field(default=None, description="Book description")
    isbn: str | None = Field(default=None, description="ISBN if available")
    open_library_key: str | None = Field(default=None, description="Open Library work key")


class BookSearchResult(BaseModel):
    """Result from Open Library book search."""

    query_author: str = Field(description="Author searched for")
    query_title: str | None = Field(default=None, description="Title searched for")
    books: list[BookInfo] = Field(default_factory=list, description="Books found")
    total_found: int = Field(default=0, description="Total books matching query")
    no_results: bool = Field(default=False, description="Flag if no results found")


class BookAnalysisReport(BaseModel):
    """Structured report for book analysis queries."""

    type: Literal["BOOK_ANALYSIS"] = Field(default="BOOK_ANALYSIS")
    query: str = Field(description="Original user query")
    books_found: list[BookInfo] = Field(default_factory=list, description="Books found")
    theme_summary: str = Field(description="Summary of common themes")
    common_themes: list[str] = Field(default_factory=list, description="List of themes")
    no_results: bool = Field(default=False, description="Flag if no results found")


# =============================================================================
# NASA Tool Schemas (NASA Open APIs - NEO)
# =============================================================================


class NEOInfo(BaseModel):
    """Information about a Near Earth Object."""

    name: str = Field(description="NEO name/designation")
    neo_id: str = Field(description="NASA NEO reference ID")
    absolute_magnitude: float | None = Field(default=None, description="Absolute magnitude H")
    estimated_diameter_min_km: float | None = Field(
        default=None, description="Minimum estimated diameter in km"
    )
    estimated_diameter_max_km: float | None = Field(
        default=None, description="Maximum estimated diameter in km"
    )
    is_potentially_hazardous: bool = Field(
        default=False, description="Potentially hazardous asteroid flag"
    )
    close_approach_date: str = Field(description="Date of close approach")
    relative_velocity_kph: float | None = Field(
        default=None, description="Relative velocity in km/h"
    )
    miss_distance_km: float | None = Field(
        default=None, description="Miss distance in kilometers"
    )
    orbiting_body: str = Field(default="Earth", description="Body being orbited")


class NEOFeedResult(BaseModel):
    """Result from NASA NEO feed API."""

    start_date: str = Field(description="Feed start date")
    end_date: str = Field(description="Feed end date")
    element_count: int = Field(default=0, description="Total NEOs in date range")
    near_earth_objects: list[NEOInfo] = Field(
        default_factory=list, description="List of NEOs"
    )


class NEOReport(BaseModel):
    """Structured report for NEO queries."""

    type: Literal["NEO_REPORT"] = Field(default="NEO_REPORT")
    query: str = Field(description="Original user query")
    date_range: str = Field(description="Date range queried")
    objects_count: int = Field(default=0, description="Total NEOs found")
    hazardous_count: int = Field(default=0, description="Potentially hazardous NEOs")
    closest_approach: NEOInfo | None = Field(
        default=None, description="NEO with closest approach"
    )
    risk_level: Literal["normal", "elevated"] = Field(
        default="normal", description="Risk assessment level"
    )
    risk_assessment: str = Field(description="Human-readable risk assessment")
    near_earth_objects: list[NEOInfo] = Field(
        default_factory=list, description="All NEOs in range"
    )


# =============================================================================
# Poetry Tool Schemas (PoetryDB)
# =============================================================================


class PoemInfo(BaseModel):
    """Information about a single poem."""

    title: str = Field(description="Poem title")
    author: str = Field(description="Poet name")
    lines: list[str] = Field(default_factory=list, description="Lines of the poem")
    linecount: int = Field(default=0, description="Number of lines")


class PoetryResult(BaseModel):
    """Result from PoetryDB search."""

    query_author: str | None = Field(default=None, description="Author searched for")
    query_title: str | None = Field(default=None, description="Title searched for")
    poems: list[PoemInfo] = Field(default_factory=list, description="Poems found")
    total_found: int = Field(default=0, description="Total poems found")
    no_results: bool = Field(default=False, description="Flag if no results found")


class PoetryAnalysisReport(BaseModel):
    """Structured report for poetry analysis queries."""

    type: Literal["POETRY_ANALYSIS"] = Field(default="POETRY_ANALYSIS")
    query: str = Field(description="Original user query")
    poem: PoemInfo | None = Field(default=None, description="Selected poem for analysis")
    focus_lines: list[str] = Field(
        default_factory=list, description="Lines being analyzed (e.g., quatrain)"
    )
    analysis: str = Field(description="Literary analysis")
    literary_devices: list[str] = Field(
        default_factory=list, description="Literary devices identified"
    )
    no_results: bool = Field(default=False, description="Flag if no poems found")


# =============================================================================
# Nutrition Tool Schemas (LogMeal Food AI)
# =============================================================================


class NutrientInfo(BaseModel):
    """Nutritional information."""

    calories: float | None = Field(default=None, description="Calories")
    protein_g: float | None = Field(default=None, description="Protein in grams")
    carbs_g: float | None = Field(default=None, description="Carbohydrates in grams")
    fat_g: float | None = Field(default=None, description="Fat in grams")
    fiber_g: float | None = Field(default=None, description="Fiber in grams")


class MealInfo(BaseModel):
    """Information about a meal recommendation."""

    name: str = Field(description="Meal name")
    description: str = Field(description="Meal description")
    ingredients: list[str] = Field(default_factory=list, description="List of ingredients")
    nutrients: NutrientInfo | None = Field(default=None, description="Nutritional info")
    diet_compatible: bool = Field(default=True, description="Compatible with requested diet")
    restriction_safe: bool = Field(
        default=True, description="Safe for dietary restrictions"
    )


class MealRecommendation(BaseModel):
    """Result from meal recommendation query."""

    diet_type: str = Field(description="Diet type requested")
    restrictions: list[str] = Field(
        default_factory=list, description="Dietary restrictions"
    )
    meals: list[MealInfo] = Field(default_factory=list, description="Recommended meals")
    conflict_detected: bool = Field(
        default=False, description="Conflicting restrictions detected"
    )
    conflict_message: str | None = Field(
        default=None, description="Description of conflict if any"
    )


class MealRecommendationReport(BaseModel):
    """Structured report for meal recommendation queries."""

    type: Literal["MEAL_RECOMMENDATION"] = Field(default="MEAL_RECOMMENDATION")
    query: str = Field(description="Original user query")
    diet_type: str = Field(description="Diet type")
    restrictions: list[str] = Field(default_factory=list, description="Restrictions applied")
    recommendations: list[MealInfo] = Field(
        default_factory=list, description="Meal recommendations"
    )
    nutritional_summary: str = Field(description="Summary of nutritional information")
    conflict_detected: bool = Field(default=False, description="Conflict in restrictions")
    conflict_message: str | None = Field(default=None, description="Conflict details")


# =============================================================================
# Agent Response Schemas
# =============================================================================


class ToolCall(BaseModel):
    """Record of a tool call made by the agent."""

    tool_name: str = Field(description="Name of tool called")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters passed to tool"
    )
    result: dict[str, Any] | None = Field(default=None, description="Tool result")
    latency_ms: float | None = Field(default=None, description="Call latency in ms")
    success: bool = Field(default=True, description="Whether call succeeded")
    error_message: str | None = Field(default=None, description="Error if failed")


# Union type for all report types
ReportType = BookAnalysisReport | NEOReport | PoetryAnalysisReport | MealRecommendationReport


class AgentResponse(BaseModel):
    """Complete agent response including tool calls and final output."""

    query: str = Field(description="Original user query")
    intent: str = Field(description="Detected intent")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tools called during processing"
    )
    report: ReportType = Field(description="Structured output report")
    reasoning: str = Field(description="Agent's reasoning process")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    total_latency_ms: float | None = Field(
        default=None, description="Total processing time"
    )


# =============================================================================
# Evaluation Schemas
# =============================================================================


class EvaluationScore(BaseModel):
    """Score from a single evaluation metric."""

    metric_name: str = Field(description="Name of the metric")
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1")
    passed: bool = Field(description="Whether score meets threshold")
    threshold: float = Field(description="Threshold for passing")
    reason: str | None = Field(default=None, description="Explanation for score")


class EvaluationResult(BaseModel):
    """Complete evaluation result for a test case."""

    test_case_id: str = Field(description="Unique test case identifier")
    input_query: str = Field(description="Input query tested")
    actual_output: str = Field(description="Actual output from agent")
    expected_output: str | None = Field(default=None, description="Expected output")
    context: list[str] = Field(
        default_factory=list, description="Context/retrieval context"
    )
    api_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw API results from tool calls for human review and debugging"
    )
    deepeval_scores: list[EvaluationScore] = Field(
        default_factory=list, description="DeepEval metric scores"
    )
    phoenix_scores: list[EvaluationScore] = Field(
        default_factory=list, description="Phoenix metric scores"
    )
    overall_passed: bool = Field(description="Whether all metrics passed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Evaluation timestamp"
    )


class ValidationError(BaseModel):
    """Error response for validation failures."""

    type: Literal["VALIDATION_ERROR"] = Field(default="VALIDATION_ERROR")
    error_code: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")
    field: str | None = Field(default=None, description="Field that caused error")
    suggestion: str | None = Field(default=None, description="Suggestion to fix")
