"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from models.schemas import BookInfo, NEOInfo, PoemInfo, MealInfo, NutrientInfo  # noqa: E402


def make_book_info(
    title: str,
    author: str,
    year: int | None = None,
    subjects: list[str] | None = None,
) -> BookInfo:
    """Factory for creating BookInfo test fixtures."""
    return BookInfo(
        title=title,
        author=author,
        year=year,
        subjects=subjects or [],
    )


def make_neo_info(
    name: str,
    neo_id: str,
    hazardous: bool = False,
    date: str = "2024-03-01",
    distance_km: float = 1000000.0,
) -> NEOInfo:
    """Factory for creating NEOInfo test fixtures."""
    return NEOInfo(
        name=name,
        neo_id=neo_id,
        is_potentially_hazardous=hazardous,
        close_approach_date=date,
        miss_distance_km=distance_km,
    )


def make_poem_info(
    title: str,
    author: str,
    lines: list[str] | None = None,
    linecount: int | None = None,
) -> PoemInfo:
    """Factory for creating PoemInfo test fixtures."""
    lines = lines or ["Sample line"] * 14
    return PoemInfo(
        title=title,
        author=author,
        lines=lines,
        linecount=linecount if linecount is not None else len(lines),
    )


def make_meal_info(
    name: str,
    description: str,
    ingredients: list[str],
    calories: float = 400.0,
    protein: float = 20.0,
    carbs: float = 30.0,
    fat: float = 15.0,
    diet_compatible: bool = True,
    restriction_safe: bool = True,
) -> MealInfo:
    """Factory for creating MealInfo test fixtures."""
    return MealInfo(
        name=name,
        description=description,
        ingredients=ingredients,
        nutrients=NutrientInfo(
            calories=calories,
            protein_g=protein,
            carbs_g=carbs,
            fat_g=fat,
        ),
        diet_compatible=diet_compatible,
        restriction_safe=restriction_safe,
    )


@pytest.fixture
def sample_book_query() -> str:
    """Sample book query for testing."""
    return "Find all books written by George Orwell published before 1950"


@pytest.fixture
def sample_neo_query() -> str:
    """Sample NEO query for testing."""
    return "Check if there are any Near Earth Objects passing by Earth this weekend"


@pytest.fixture
def sample_poetry_query() -> str:
    """Sample poetry query for testing."""
    return "Find a sonnet by William Shakespeare and explain the metaphor"


@pytest.fixture
def sample_nutrition_query() -> str:
    """Sample nutrition query for testing."""
    return "Based on a Mediterranean diet, recommend three dinner options that avoid dairy"


@pytest.fixture
def test_cases() -> list[dict[str, str | list[str]]]:
    """Standard test cases for evaluation."""
    return [
        {
            "test_case_id": "book_001",
            "input_query": "Find all books by George Orwell published before 1950",
            "expected_tools": ["book_search"],
        },
        {
            "test_case_id": "neo_001",
            "input_query": "Check NEO data for this weekend",
            "expected_tools": ["nasa_neo"],
        },
        {
            "test_case_id": "poetry_001",
            "input_query": "Find a sonnet by Shakespeare",
            "expected_tools": ["poetry_search"],
        },
        {
            "test_case_id": "nutrition_001",
            "input_query": "Recommend Mediterranean meals without dairy",
            "expected_tools": ["nutrition_meal_recommendation"],
        },
        {
            "test_case_id": "edge_001",
            "input_query": "Check NEO data for February 30th",
            "expected_tools": ["nasa_neo"],
        },
    ]
