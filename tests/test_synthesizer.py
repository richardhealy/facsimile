"""Tests for M3 vision synthesis: Answer, Synthesizer protocol, StubSynthesizer."""

from pathlib import Path

import pytest

from facsimile.answer.synthesizer import (
    Answer,
    StubSynthesizer,
    Synthesizer,
    _extract_cited_pages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pages(nums: list[int]) -> list[tuple[int, Path]]:
    return [(n, Path(f"/tmp/page_{n}.png")) for n in nums]


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------

class TestAnswer:
    def test_has_text_field(self):
        a = Answer(text="hello", cited_pages=[1, 3], model="stub")
        assert a.text == "hello"

    def test_has_cited_pages_field(self):
        a = Answer(text="hello", cited_pages=[2, 5], model="stub")
        assert a.cited_pages == [2, 5]

    def test_has_model_field(self):
        a = Answer(text="hello", cited_pages=[], model="my-model")
        assert a.model == "my-model"

    def test_equality(self):
        a = Answer(text="hi", cited_pages=[1], model="stub")
        b = Answer(text="hi", cited_pages=[1], model="stub")
        assert a == b

    def test_empty_cited_pages_allowed(self):
        a = Answer(text="nothing found", cited_pages=[], model="stub")
        assert a.cited_pages == []


# ---------------------------------------------------------------------------
# StubSynthesizer
# ---------------------------------------------------------------------------

class TestStubSynthesizer:
    def test_returns_answer_instance(self):
        result = StubSynthesizer().answer("What is revenue?", _pages([1, 2]))
        assert isinstance(result, Answer)

    def test_cites_all_provided_pages(self):
        result = StubSynthesizer().answer("question", _pages([3, 7]))
        assert 3 in result.cited_pages
        assert 7 in result.cited_pages

    def test_cited_pages_sorted(self):
        result = StubSynthesizer().answer("q", _pages([5, 1, 3]))
        assert result.cited_pages == sorted(result.cited_pages)

    def test_no_duplicate_pages(self):
        result = StubSynthesizer().answer("q", _pages([2, 2, 2]))
        assert result.cited_pages == [2]

    def test_model_reported(self):
        result = StubSynthesizer(model="test-model").answer("q", _pages([1]))
        assert result.model == "test-model"

    def test_default_model_is_stub(self):
        result = StubSynthesizer().answer("q", _pages([1]))
        assert result.model == "stub"

    def test_empty_pages_returns_answer(self):
        result = StubSynthesizer().answer("anything", [])
        assert isinstance(result, Answer)
        assert result.cited_pages == []

    def test_text_contains_question(self):
        q = "What is the EBITDA margin?"
        result = StubSynthesizer().answer(q, _pages([1]))
        assert q in result.text

    def test_text_contains_page_number(self):
        result = StubSynthesizer().answer("q", _pages([42]))
        assert "42" in result.text

    def test_single_page(self):
        result = StubSynthesizer().answer("q", _pages([1]))
        assert result.cited_pages == [1]


# ---------------------------------------------------------------------------
# Synthesizer protocol compliance
# ---------------------------------------------------------------------------

def test_stub_satisfies_synthesizer_protocol():
    assert isinstance(StubSynthesizer(), Synthesizer)


# ---------------------------------------------------------------------------
# _extract_cited_pages helper
# ---------------------------------------------------------------------------

class TestExtractCitedPages:
    def test_extracts_single_page(self):
        assert _extract_cited_pages("According to page 3, revenue is up.", {3, 5}) == [3]

    def test_extracts_multiple_pages(self):
        text = "See page 1 and page 5 for details."
        assert _extract_cited_pages(text, {1, 5, 7}) == [1, 5]

    def test_ignores_unavailable_pages(self):
        text = "According to page 99, something happened."
        assert _extract_cited_pages(text, {1, 2, 3}) == []

    def test_case_insensitive(self):
        text = "PAGE 2 shows the chart."
        assert _extract_cited_pages(text, {2}) == [2]

    def test_returns_sorted_list(self):
        text = "See page 5 and page 2."
        assert _extract_cited_pages(text, {2, 5}) == [2, 5]

    def test_empty_text_returns_empty(self):
        assert _extract_cited_pages("", {1, 2}) == []

    def test_no_page_mentions_returns_empty(self):
        assert _extract_cited_pages("No citations here at all.", {1, 2}) == []

    def test_deduplicates_same_page_mentioned_twice(self):
        text = "As shown on page 4 and again on page 4."
        assert _extract_cited_pages(text, {4}) == [4]

    def test_empty_available_returns_empty(self):
        text = "See page 1."
        assert _extract_cited_pages(text, set()) == []


# ---------------------------------------------------------------------------
# ClaudeSynthesizer import guard (no API key needed)
# ---------------------------------------------------------------------------

def test_claude_synthesizer_requires_anthropic_package():
    """ClaudeSynthesizer raises ImportError when anthropic is not installed."""
    import sys
    import importlib
    # Temporarily hide the anthropic package if installed.
    anthropic_mod = sys.modules.pop("anthropic", None)
    try:
        from facsimile.answer.synthesizer import ClaudeSynthesizer
        with pytest.raises(ImportError, match="anthropic"):
            ClaudeSynthesizer()
    finally:
        if anthropic_mod is not None:
            sys.modules["anthropic"] = anthropic_mod
