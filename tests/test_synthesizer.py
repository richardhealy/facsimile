"""Tests for M3 vision synthesis: Answer, StubSynthesizer, Synthesizer protocol."""

import sys
from pathlib import Path

import pytest

from facsimile.answer.synthesizer import Answer, StubSynthesizer, Synthesizer


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_text_field(self):
        a = Answer(text="Revenue was $10M.", cited_pages=[3], model="stub")
        assert a.text == "Revenue was $10M."

    def test_cited_pages_field(self):
        a = Answer(text="hi", cited_pages=[2, 5], model="stub")
        assert a.cited_pages == [2, 5]

    def test_model_field(self):
        a = Answer(text="hi", cited_pages=[], model="claude-opus-4-8")
        assert a.model == "claude-opus-4-8"

    def test_empty_cited_pages(self):
        a = Answer(text="nothing found", cited_pages=[], model="stub")
        assert a.cited_pages == []


# ---------------------------------------------------------------------------
# StubSynthesizer
# ---------------------------------------------------------------------------


class TestStubSynthesizer:
    def test_returns_answer_instance(self):
        stub = StubSynthesizer()
        result = stub.synthesize("what is revenue?", [(1, Path("/tmp/p1.png"))])
        assert isinstance(result, Answer)

    def test_cited_pages_contains_all_input_pages(self):
        stub = StubSynthesizer()
        pages = [(3, Path("/tmp/p3.png")), (7, Path("/tmp/p7.png")), (12, Path("/tmp/p12.png"))]
        result = stub.synthesize("test question", pages)
        assert 3 in result.cited_pages
        assert 7 in result.cited_pages
        assert 12 in result.cited_pages

    def test_cited_pages_order_matches_input(self):
        stub = StubSynthesizer()
        pages = [(1, Path("/tmp/p1.png")), (4, Path("/tmp/p4.png"))]
        result = stub.synthesize("q", pages)
        assert result.cited_pages == [1, 4]

    def test_text_contains_question(self):
        stub = StubSynthesizer()
        result = stub.synthesize("what are total assets?", [(2, Path("/tmp/p2.png"))])
        assert "what are total assets?" in result.text

    def test_empty_page_list_returns_empty_cited(self):
        stub = StubSynthesizer()
        result = stub.synthesize("no pages", [])
        assert result.cited_pages == []

    def test_model_default_is_stub(self):
        stub = StubSynthesizer()
        result = stub.synthesize("q", [(1, Path("/tmp/p1.png"))])
        assert result.model == "stub"

    def test_custom_model_stored(self):
        stub = StubSynthesizer(model="my-test-model")
        result = stub.synthesize("q", [(1, Path("/tmp/p1.png"))])
        assert result.model == "my-test-model"

    def test_single_page_cited(self):
        stub = StubSynthesizer()
        result = stub.synthesize("q", [(5, Path("/tmp/p5.png"))])
        assert result.cited_pages == [5]

    def test_text_is_non_empty(self):
        stub = StubSynthesizer()
        result = stub.synthesize("q", [(1, Path("/tmp/p1.png"))])
        assert len(result.text) > 0


# ---------------------------------------------------------------------------
# Synthesizer protocol compliance
# ---------------------------------------------------------------------------


def test_stub_synthesizer_satisfies_protocol():
    assert isinstance(StubSynthesizer(), Synthesizer)


def test_protocol_is_runtime_checkable():
    class NotASynthesizer:
        pass

    assert not isinstance(NotASynthesizer(), Synthesizer)


# ---------------------------------------------------------------------------
# ClaudeSynthesizer — import-error guard (no API key required)
# ---------------------------------------------------------------------------


def test_claude_synthesizer_raises_import_error_without_anthropic(monkeypatch):
    """Simulates anthropic not installed by injecting None into sys.modules."""
    monkeypatch.setitem(sys.modules, "anthropic", None)
    # Re-importing after monkeypatch ensures __init__ sees the patched modules dict.
    from facsimile.answer.synthesizer import ClaudeSynthesizer

    with pytest.raises(ImportError, match="anthropic"):
        ClaudeSynthesizer()
