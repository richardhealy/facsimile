"""Benchmark runner: visual vs text pipeline on table/chart questions.

The public entry points are :func:`run_benchmark` and :func:`summarize_results`.
The runner evaluates each :class:`~facsimile.bench.questions.BenchQuestion`
through two pipelines and records per-question results as :class:`BenchResult`
objects.

**Visual pipeline** (facsimile):
  1. ``Retriever.retrieve()`` — late-interaction MaxSim over indexed pages.
  2. Page rasterization → ``synthesizer_fn()`` (vision-LLM).
  3. A result is *correct* when the expected-answer substring appears in the
     synthesized text (case-insensitive).
  4. A page is a *hit* when ``correct_page`` appears in the cited-page list.

**Text pipeline** (baseline):
  1. :func:`~facsimile.bench.text_pipeline.text_retrieve` — keyword overlap.
  2. Concatenated text from the top-k pages acts as the "answer".
  3. A result is *correct* when the expected-answer substring appears in the
     concatenated text.
  4. A page is a *hit* when ``correct_page`` is in the retrieved-page list.

Aggregate metrics are computed by :func:`summarize_results`, which also
breaks down accuracy by ``question_type`` (``"table"`` / ``"chart"`` /
``"text"``).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from facsimile.answer.synthesizer import Answer, synthesize
from facsimile.bench.questions import BenchQuestion
from facsimile.bench.text_pipeline import text_retrieve
from facsimile.ingest.rasterize import rasterize_pdf
from facsimile.retrieve.retriever import Retriever


@dataclass
class BenchResult:
    """Per-question result from :func:`run_benchmark`.

    Attributes:
        question: The original question text.
        expected_answer: Expected-answer substring used for correctness checks.
        correct_page: The ground-truth page number.
        question_type: Category tag (``"table"`` / ``"chart"`` / ``"text"``).
        visual_answer: Synthesized answer text from the vision-LLM pipeline.
        visual_cited_pages: Page numbers cited by the vision model.
        visual_correct: ``True`` when *expected_answer* appears in *visual_answer*.
        visual_page_hit: ``True`` when *correct_page* is in *visual_cited_pages*.
        text_answer: Concatenated text from the top-k pages of the text pipeline.
        text_retrieved_pages: Page numbers returned by the text pipeline.
        text_correct: ``True`` when *expected_answer* appears in *text_answer*.
        text_page_hit: ``True`` when *correct_page* is in *text_retrieved_pages*.
    """

    question: str
    expected_answer: str
    correct_page: int
    question_type: str

    # Visual pipeline
    visual_answer: str
    visual_cited_pages: list[int]
    visual_correct: bool
    visual_page_hit: bool

    # Text pipeline
    text_answer: str
    text_retrieved_pages: list[int]
    text_correct: bool
    text_page_hit: bool


@dataclass
class BenchSummary:
    """Aggregate accuracy metrics from :func:`summarize_results`.

    Attributes:
        n: Total number of questions evaluated.
        visual_answer_accuracy: Fraction of questions where the visual pipeline
            produced a correct answer.
        text_answer_accuracy: Fraction of questions where the text pipeline
            produced a correct answer.
        visual_page_accuracy: Fraction of questions where the visual pipeline
            cited the correct page.
        text_page_accuracy: Fraction of questions where the text pipeline
            retrieved the correct page.
        by_type: Per question-type breakdown; keys are question-type strings
            (``"table"``, ``"chart"``, ``"text"``), values are dicts with the
            same four accuracy keys plus ``"n"``.
    """

    n: int
    visual_answer_accuracy: float
    text_answer_accuracy: float
    visual_page_accuracy: float
    text_page_accuracy: float
    by_type: dict[str, dict] = field(default_factory=dict)


def _contains(answer_text: str, expected: str) -> bool:
    return expected.lower() in answer_text.lower()


def run_benchmark(
    questions: list[BenchQuestion],
    pdf_path: Path,
    retriever: Retriever,
    synthesizer_fn: Callable = synthesize,
    top_k: int = 3,
    rasterize_dpi: int = 150,
) -> list[BenchResult]:
    """Evaluate *questions* against both the visual and text pipelines.

    The PDF is rasterized once into a temporary directory; visual results use
    the pre-rendered PNGs so rasterization cost is paid only once.

    Args:
        questions: Benchmark question set.
        pdf_path: Source PDF (must already be indexed in the Retriever's store).
        retriever: Configured :class:`~facsimile.retrieve.retriever.Retriever`
            whose index contains the pages of *pdf_path*.
        synthesizer_fn: Vision-synthesis callable; defaults to
            :func:`~facsimile.answer.synthesizer.synthesize`.  Swap for a mock
            in unit tests.
        top_k: Number of pages to retrieve per pipeline.
        rasterize_dpi: DPI for page rasterization used by the visual pipeline.

    Returns:
        One :class:`BenchResult` per question.
    """
    if not questions:
        return []

    pdf_path = Path(pdf_path)
    results: list[BenchResult] = []

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)

        # Rasterize all pages once; keep a page_num → PNG-path map.
        page_images_pil = rasterize_pdf(pdf_path, dpi=rasterize_dpi)
        page_png: dict[int, Path] = {}
        for i, img in enumerate(page_images_pil):
            page_num = i + 1
            img_path = tmp / f"page_{page_num:04d}.png"
            img.save(img_path, format="PNG")
            page_png[page_num] = img_path

        for q in questions:
            # ---- Visual pipeline ----
            retrieved = retriever.retrieve(q.question, top_k=top_k)
            vis_inputs = [
                (p.page_num, page_png[p.page_num])
                for p in retrieved
                if p.page_num in page_png
            ]

            if vis_inputs:
                answer: Answer = synthesizer_fn(q.question, vis_inputs)
                visual_answer = answer.text
                visual_cited_pages = answer.cited_pages
            else:
                visual_answer = ""
                visual_cited_pages = []

            visual_correct = _contains(visual_answer, q.expected_answer)
            visual_page_hit = q.correct_page in visual_cited_pages

            # ---- Text pipeline ----
            text_pages = text_retrieve(pdf_path, q.question, top_k=top_k)
            text_retrieved_pages = [p.page_num for p in text_pages]
            text_combined = " ".join(p.text for p in text_pages)
            text_correct = _contains(text_combined, q.expected_answer)
            text_page_hit = q.correct_page in text_retrieved_pages

            results.append(
                BenchResult(
                    question=q.question,
                    expected_answer=q.expected_answer,
                    correct_page=q.correct_page,
                    question_type=q.question_type,
                    visual_answer=visual_answer,
                    visual_cited_pages=visual_cited_pages,
                    visual_correct=visual_correct,
                    visual_page_hit=visual_page_hit,
                    text_answer=text_combined,
                    text_retrieved_pages=text_retrieved_pages,
                    text_correct=text_correct,
                    text_page_hit=text_page_hit,
                )
            )

    return results


def summarize_results(results: list[BenchResult]) -> BenchSummary:
    """Aggregate per-question results into overall and per-type accuracy metrics.

    Args:
        results: Output of :func:`run_benchmark`.

    Returns:
        :class:`BenchSummary` with ``visual_answer_accuracy``,
        ``text_answer_accuracy``, ``visual_page_accuracy``,
        ``text_page_accuracy``, and a ``by_type`` breakdown.
    """
    if not results:
        return BenchSummary(
            n=0,
            visual_answer_accuracy=0.0,
            text_answer_accuracy=0.0,
            visual_page_accuracy=0.0,
            text_page_accuracy=0.0,
        )

    n = len(results)

    by_type: dict[str, list[BenchResult]] = {}
    for r in results:
        by_type.setdefault(r.question_type, []).append(r)

    type_stats: dict[str, dict] = {}
    for qtype, items in by_type.items():
        ni = len(items)
        type_stats[qtype] = {
            "n": ni,
            "visual_answer_accuracy": sum(r.visual_correct for r in items) / ni,
            "text_answer_accuracy": sum(r.text_correct for r in items) / ni,
            "visual_page_accuracy": sum(r.visual_page_hit for r in items) / ni,
            "text_page_accuracy": sum(r.text_page_hit for r in items) / ni,
        }

    return BenchSummary(
        n=n,
        visual_answer_accuracy=sum(r.visual_correct for r in results) / n,
        text_answer_accuracy=sum(r.text_correct for r in results) / n,
        visual_page_accuracy=sum(r.visual_page_hit for r in results) / n,
        text_page_accuracy=sum(r.text_page_hit for r in results) / n,
        by_type=type_stats,
    )
