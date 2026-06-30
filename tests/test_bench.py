"""Tests for M4 benchmark: questions, text pipeline, runner, and summarize."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pypdfium2 as pdfium
import pytest

from facsimile.bench.questions import BenchQuestion, load_questions
from facsimile.bench.runner import BenchResult, BenchSummary, run_benchmark, summarize_results
from facsimile.bench.text_pipeline import (
    TextPage,
    _score,
    _tokenize,
    extract_page_texts,
    text_retrieve,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_blank_pdf(n_pages: int = 2) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(width=612, height=792)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _write_pdf(tmp_path: Path, n_pages: int = 2) -> Path:
    pdf = _make_blank_pdf(n_pages)
    p = tmp_path / "test.pdf"
    p.write_bytes(pdf)
    return p


def _make_result(
    *,
    question_type: str = "table",
    visual_correct: bool = True,
    visual_page_hit: bool = True,
    text_correct: bool = False,
    text_page_hit: bool = False,
) -> BenchResult:
    return BenchResult(
        question="What is revenue?",
        expected_answer="revenue",
        correct_page=2,
        question_type=question_type,
        visual_answer="Revenue is 42M [Page 2]",
        visual_cited_pages=[2],
        visual_correct=visual_correct,
        visual_page_hit=visual_page_hit,
        text_answer="some unrelated text",
        text_retrieved_pages=[1, 3],
        text_correct=text_correct,
        text_page_hit=text_page_hit,
    )


# ---------------------------------------------------------------------------
# BenchQuestion
# ---------------------------------------------------------------------------


class TestBenchQuestion:
    def test_fields(self):
        q = BenchQuestion(
            question="What is revenue?",
            expected_answer="42M",
            correct_page=3,
            question_type="table",
        )
        assert q.question == "What is revenue?"
        assert q.expected_answer == "42M"
        assert q.correct_page == 3
        assert q.question_type == "table"

    def test_default_question_type(self):
        q = BenchQuestion(question="q", expected_answer="a", correct_page=1)
        assert q.question_type == "table"


# ---------------------------------------------------------------------------
# load_questions
# ---------------------------------------------------------------------------


class TestLoadQuestions:
    def test_loads_list(self, tmp_path):
        data = [
            {"question": "q1", "expected_answer": "a1", "correct_page": 1, "question_type": "table"},
            {"question": "q2", "expected_answer": "a2", "correct_page": 2, "question_type": "chart"},
        ]
        p = tmp_path / "qs.json"
        p.write_text(json.dumps(data))
        qs = load_questions(p)
        assert len(qs) == 2
        assert qs[0].question == "q1"
        assert qs[1].question_type == "chart"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "nope.json")

    def test_not_array_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"question": "q"}')
        with pytest.raises(ValueError, match="array"):
            load_questions(p)

    def test_default_question_type_in_json(self, tmp_path):
        data = [{"question": "q", "expected_answer": "a", "correct_page": 1}]
        p = tmp_path / "qs.json"
        p.write_text(json.dumps(data))
        qs = load_questions(p)
        assert qs[0].question_type == "table"


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lower_cases(self):
        tokens = _tokenize("Revenue GROWTH")
        assert "revenue" in tokens
        assert "growth" in tokens

    def test_drops_stop_words(self):
        tokens = _tokenize("what is the revenue")
        assert "revenue" in tokens
        assert "what" not in tokens
        assert "is" not in tokens
        assert "the" not in tokens

    def test_splits_non_alpha(self):
        tokens = _tokenize("net-income: 42.5M")
        assert "net" in tokens
        assert "income" in tokens
        assert "42" in tokens
        assert "5m" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_all_stopwords(self):
        assert _tokenize("what is the") == []


# ---------------------------------------------------------------------------
# _score
# ---------------------------------------------------------------------------


class TestScore:
    def test_perfect_match(self):
        s = _score("revenue growth", "revenue growth rate")
        assert s == 1.0

    def test_partial_match(self):
        s = _score("revenue growth profit", "only revenue here")
        assert 0 < s < 1.0

    def test_no_match(self):
        s = _score("revenue growth", "something completely unrelated xyz")
        assert s == 0.0

    def test_empty_query(self):
        assert _score("", "some text") == 0.0

    def test_empty_query_stopwords_only(self):
        assert _score("what is the", "any text here") == 0.0

    def test_case_insensitive(self):
        s1 = _score("Revenue", "revenue growth")
        s2 = _score("revenue", "Revenue growth")
        assert s1 == s2 == 1.0


# ---------------------------------------------------------------------------
# extract_page_texts
# ---------------------------------------------------------------------------


class TestExtractPageTexts:
    def test_returns_one_per_page(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=3)
        results = extract_page_texts(pdf_path)
        assert len(results) == 3

    def test_page_numbers_one_based(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=2)
        results = extract_page_texts(pdf_path)
        page_nums = [r[0] for r in results]
        assert page_nums == [1, 2]

    def test_blank_pages_return_empty_string(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=1)
        results = extract_page_texts(pdf_path)
        assert isinstance(results[0][1], str)

    def test_single_page(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=1)
        results = extract_page_texts(pdf_path)
        assert len(results) == 1
        assert results[0][0] == 1


# ---------------------------------------------------------------------------
# text_retrieve
# ---------------------------------------------------------------------------


class TestTextRetrieve:
    def test_respects_top_k(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=5)
        results = text_retrieve(pdf_path, "revenue", top_k=3)
        assert len(results) <= 3

    def test_returns_text_page_objects(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=2)
        results = text_retrieve(pdf_path, "revenue")
        assert all(isinstance(r, TextPage) for r in results)

    def test_scores_descending(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=4)
        results = text_retrieve(pdf_path, "revenue", top_k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_mocked_text_retrieval_finds_keyword_page(self, tmp_path):
        """Patch extract_page_texts to verify ranking logic."""
        pdf_path = _write_pdf(tmp_path, n_pages=3)
        fake_texts = [
            (1, "Operating costs and overhead"),
            (2, "Revenue table: total revenue 42.5M fiscal year"),
            (3, "Board of directors names"),
        ]
        with patch("facsimile.bench.text_pipeline.extract_page_texts", return_value=fake_texts):
            results = text_retrieve(pdf_path, "total revenue fiscal year", top_k=3)
        assert results[0].page_num == 2
        assert results[0].score > 0.0

    def test_zero_score_pages_still_returned(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=2)
        fake_texts = [(1, ""), (2, "")]
        with patch("facsimile.bench.text_pipeline.extract_page_texts", return_value=fake_texts):
            results = text_retrieve(pdf_path, "revenue", top_k=2)
        assert len(results) == 2
        assert all(r.score == 0.0 for r in results)


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


def _fake_retriever(page_nums: list[int]) -> MagicMock:
    """Return a mock Retriever whose .retrieve() returns pages with given nums."""
    from facsimile.retrieve.retriever import RetrievedPage

    mock = MagicMock()
    mock.retrieve.return_value = [
        RetrievedPage(
            doc_id="test",
            page_num=pn,
            doc_path="/tmp/test.pdf",
            score=1.0,
        )
        for pn in page_nums
    ]
    return mock


def _fake_synthesizer(answer_text: str, cited: list[int]):
    from facsimile.answer.synthesizer import Answer

    def fn(question, page_images, **kwargs):
        return Answer(text=answer_text, cited_pages=cited, model="stub")

    return fn


class TestRunBenchmark:
    def test_empty_questions_returns_empty(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=1)
        retriever = _fake_retriever([1])
        results = run_benchmark([], pdf_path, retriever)
        assert results == []

    def test_returns_one_result_per_question(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=3)
        questions = [
            BenchQuestion("What is revenue?", "42", 1, "table"),
            BenchQuestion("What is profit?", "10", 2, "table"),
        ]
        retriever = _fake_retriever([1, 2])
        synthesizer = _fake_synthesizer("Answer is 42 [Page 1]", [1])
        results = run_benchmark(questions, pdf_path, retriever, synthesizer_fn=synthesizer)
        assert len(results) == 2

    def test_result_fields_populated(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=2)
        q = BenchQuestion("Revenue?", "revenue", 1, "table")
        retriever = _fake_retriever([1])
        synthesizer = _fake_synthesizer("The revenue is 50M [Page 1]", [1])
        results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        r = results[0]
        assert r.question == "Revenue?"
        assert r.expected_answer == "revenue"
        assert r.correct_page == 1
        assert r.question_type == "table"
        assert r.visual_answer == "The revenue is 50M [Page 1]"
        assert r.visual_cited_pages == [1]

    def test_visual_correct_when_answer_contains_expected(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=2)
        q = BenchQuestion("Revenue?", "42", 1, "table")
        retriever = _fake_retriever([1])
        synthesizer = _fake_synthesizer("Revenue is 42M [Page 1]", [1])
        results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        assert results[0].visual_correct is True

    def test_visual_incorrect_when_answer_lacks_expected(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=2)
        q = BenchQuestion("Revenue?", "xyz_not_present", 1, "table")
        retriever = _fake_retriever([1])
        synthesizer = _fake_synthesizer("Revenue is 42M [Page 1]", [1])
        results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        assert results[0].visual_correct is False

    def test_visual_page_hit_when_correct_page_cited(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=3)
        q = BenchQuestion("Revenue?", "42", 2, "table")
        retriever = _fake_retriever([1, 2])
        synthesizer = _fake_synthesizer("Revenue [Page 2]", [2])
        results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        assert results[0].visual_page_hit is True

    def test_visual_page_miss_when_correct_page_not_cited(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=3)
        q = BenchQuestion("Revenue?", "42", 3, "table")
        retriever = _fake_retriever([1, 2])
        synthesizer = _fake_synthesizer("Info [Page 1]", [1])
        results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        assert results[0].visual_page_hit is False

    def test_text_fields_populated(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=2)
        q = BenchQuestion("Revenue?", "42", 1, "table")
        retriever = _fake_retriever([1])
        synthesizer = _fake_synthesizer("ok", [1])
        with patch("facsimile.bench.runner.text_retrieve") as mock_tr:
            mock_tr.return_value = [TextPage(page_num=1, text="Revenue 42M", score=1.0)]
            results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        r = results[0]
        assert r.text_retrieved_pages == [1]
        assert "Revenue 42M" in r.text_answer

    def test_text_correct_when_expected_in_page_text(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=2)
        q = BenchQuestion("Revenue?", "42M", 1, "table")
        retriever = _fake_retriever([1])
        synthesizer = _fake_synthesizer("ok", [1])
        with patch("facsimile.bench.runner.text_retrieve") as mock_tr:
            mock_tr.return_value = [TextPage(page_num=1, text="Revenue is 42M", score=1.0)]
            results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        assert results[0].text_correct is True

    def test_text_page_hit_when_correct_page_retrieved(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=3)
        q = BenchQuestion("Revenue?", "42", 2, "table")
        retriever = _fake_retriever([1])
        synthesizer = _fake_synthesizer("ok", [1])
        with patch("facsimile.bench.runner.text_retrieve") as mock_tr:
            mock_tr.return_value = [
                TextPage(page_num=2, text="Revenue 42M", score=0.9),
                TextPage(page_num=1, text="intro", score=0.1),
            ]
            results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        assert results[0].text_page_hit is True

    def test_case_insensitive_answer_matching(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, n_pages=1)
        q = BenchQuestion("Revenue?", "REVENUE", 1, "table")
        retriever = _fake_retriever([1])
        synthesizer = _fake_synthesizer("revenue is 10M [Page 1]", [1])
        results = run_benchmark([q], pdf_path, retriever, synthesizer_fn=synthesizer)
        assert results[0].visual_correct is True


# ---------------------------------------------------------------------------
# summarize_results
# ---------------------------------------------------------------------------


class TestSummarizeResults:
    def test_empty_returns_zero_summary(self):
        summary = summarize_results([])
        assert summary.n == 0
        assert summary.visual_answer_accuracy == 0.0
        assert summary.text_answer_accuracy == 0.0
        assert summary.visual_page_accuracy == 0.0
        assert summary.text_page_accuracy == 0.0

    def test_all_correct(self):
        results = [
            _make_result(visual_correct=True, visual_page_hit=True, text_correct=True, text_page_hit=True)
            for _ in range(3)
        ]
        s = summarize_results(results)
        assert s.n == 3
        assert s.visual_answer_accuracy == 1.0
        assert s.text_answer_accuracy == 1.0
        assert s.visual_page_accuracy == 1.0
        assert s.text_page_accuracy == 1.0

    def test_all_incorrect(self):
        results = [
            _make_result(visual_correct=False, visual_page_hit=False, text_correct=False, text_page_hit=False)
            for _ in range(4)
        ]
        s = summarize_results(results)
        assert s.visual_answer_accuracy == 0.0
        assert s.text_answer_accuracy == 0.0

    def test_mixed_accuracy(self):
        results = [
            _make_result(visual_correct=True, text_correct=False),
            _make_result(visual_correct=True, text_correct=False),
            _make_result(visual_correct=False, text_correct=True),
            _make_result(visual_correct=False, text_correct=True),
        ]
        s = summarize_results(results)
        assert s.n == 4
        assert s.visual_answer_accuracy == 0.5
        assert s.text_answer_accuracy == 0.5

    def test_by_type_breakdown(self):
        results = [
            _make_result(question_type="table", visual_correct=True),
            _make_result(question_type="table", visual_correct=False),
            _make_result(question_type="chart", visual_correct=True),
        ]
        s = summarize_results(results)
        assert "table" in s.by_type
        assert "chart" in s.by_type
        assert s.by_type["table"]["n"] == 2
        assert s.by_type["table"]["visual_answer_accuracy"] == 0.5
        assert s.by_type["chart"]["n"] == 1
        assert s.by_type["chart"]["visual_answer_accuracy"] == 1.0

    def test_by_type_includes_all_metrics(self):
        results = [_make_result()]
        s = summarize_results(results)
        for qtype_stats in s.by_type.values():
            assert "n" in qtype_stats
            assert "visual_answer_accuracy" in qtype_stats
            assert "text_answer_accuracy" in qtype_stats
            assert "visual_page_accuracy" in qtype_stats
            assert "text_page_accuracy" in qtype_stats

    def test_page_accuracy(self):
        results = [
            _make_result(visual_page_hit=True, text_page_hit=False),
            _make_result(visual_page_hit=False, text_page_hit=True),
        ]
        s = summarize_results(results)
        assert s.visual_page_accuracy == 0.5
        assert s.text_page_accuracy == 0.5


# ---------------------------------------------------------------------------
# Public imports from facsimile.bench
# ---------------------------------------------------------------------------


class TestPublicImports:
    def test_import_from_package(self):
        from facsimile.bench import (
            BenchQuestion,
            BenchResult,
            BenchSummary,
            TextPage,
            extract_page_texts,
            load_questions,
            run_benchmark,
            summarize_results,
            text_retrieve,
        )
        assert BenchQuestion is not None
        assert run_benchmark is not None
        assert summarize_results is not None
        assert text_retrieve is not None
