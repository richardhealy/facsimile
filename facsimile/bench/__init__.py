"""Bench: table/chart question set and visual-vs-text comparison."""

from facsimile.bench.questions import BenchQuestion, load_questions
from facsimile.bench.runner import BenchResult, BenchSummary, run_benchmark, summarize_results
from facsimile.bench.text_pipeline import TextPage, extract_page_texts, text_retrieve

__all__ = [
    "BenchQuestion",
    "load_questions",
    "BenchResult",
    "BenchSummary",
    "run_benchmark",
    "summarize_results",
    "TextPage",
    "extract_page_texts",
    "text_retrieve",
]
