"""Benchmark runner: visual vs text pipeline on table/chart questions (stub for M4)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchResult:
    question: str
    expected: str
    visual_answer: str
    text_answer: str
    visual_correct: bool
    text_correct: bool


def run_benchmark(
    question_set: Path,
    pdf_path: Path,
) -> list[BenchResult]:
    """
    Run table/chart questions through both visual and text pipelines.
    Returns per-question results for comparison.
    Implemented in M4.
    """
    raise NotImplementedError("Benchmark implemented in M4")
