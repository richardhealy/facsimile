"""Benchmark question schema and loader.

A :class:`BenchQuestion` describes one evaluation item: the natural-language
question, the expected-answer substring to match, the 1-based page number where
the answer lives, and a category tag (``"table"`` / ``"chart"`` / ``"text"``).

Call :func:`load_questions` to load a question set from a JSON file whose items
match the :class:`BenchQuestion` fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchQuestion:
    """One benchmark evaluation item."""

    question: str
    expected_answer: str
    correct_page: int
    question_type: str = "table"


def load_questions(path: Path) -> list[BenchQuestion]:
    """Load a JSON question set and return a list of :class:`BenchQuestion`.

    The JSON file must be a list of objects whose keys match the
    :class:`BenchQuestion` fields.  ``question_type`` is optional and defaults
    to ``"table"``.

    Args:
        path: Path to the ``.json`` question file.

    Returns:
        Parsed list of :class:`BenchQuestion` objects.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the JSON structure is invalid.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Question file must be a JSON array")
    return [BenchQuestion(**item) for item in data]
