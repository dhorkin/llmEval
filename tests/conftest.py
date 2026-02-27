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
