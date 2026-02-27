"""Re-export test helpers from parent conftest for regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add tests directory to path so we can import from parent conftest
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: E402
    make_book_info,
    make_meal_info,
    make_neo_info,
    make_poem_info,
)

__all__ = ["make_book_info", "make_meal_info", "make_neo_info", "make_poem_info"]
