"""Tests for M3 synthesis: Answer, StubSynthesizer, ClaudeSynthesizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from facsimile.answer.synthesizer import (
    Answer,
    ClaudeSynthesizer,
    StubSynthesizer,
    Synthesizer,
    synthesize,
)


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_fields_accessible(self):
        a = Answer(text="hello", cited_pages=[1, 2], model="stub")
        assert a.text == "hello"
        assert a.cited_pages == [1, 2]
        assert a.model == "stub"

    def test_empty_citations(self):
        a = Answer(text="nothing found", cited_pages=[], model="stub")
        assert a.cited_pages == []


# ---------------------------------------------------------------------------
# StubSynthesizer
# ---------------------------------------------------------------------------


class TestStubSynthesizer:
    def test_returns_answer(self, tmp_path):
        stub = StubSynthesizer()
        result = stub.synthesize("what is revenue?", [(1, tmp_path / "p1.png")])
        assert isinstance(result, Answer)

    def test_default_model_is_stub(self):
        assert StubSynthesizer().model == "stub"

    def test_custom_model_stored(self):
        assert StubSynthesizer(model="my-model").model == "my-model"

    def test_text_contains_question(self, tmp_path):
        stub = StubSynthesizer()
        result = stub.synthesize("what is profit?", [(1, tmp_path / "p.png")])
        assert "what is profit?" in result.text

    def test_cited_pages_match_input(self, tmp_path):
        stub = StubSynthesizer()
        pages = [(3, tmp_path / "a.png"), (7, tmp_path / "b.png")]
        result = stub.synthesize("query", pages)
        assert result.cited_pages == [3, 7]

    def test_empty_pages_returns_empty_citations(self):
        stub = StubSynthesizer()
        result = stub.synthesize("query", [])
        assert result.cited_pages == []

    def test_text_cites_pages(self, tmp_path):
        stub = StubSynthesizer()
        result = stub.synthesize("q", [(2, tmp_path / "x.png"), (5, tmp_path / "y.png")])
        assert "[Page 2]" in result.text
        assert "[Page 5]" in result.text


# ---------------------------------------------------------------------------
# Synthesizer protocol compliance
# ---------------------------------------------------------------------------


def test_stub_satisfies_synthesizer_protocol():
    assert isinstance(StubSynthesizer(), Synthesizer)


# ---------------------------------------------------------------------------
# ClaudeSynthesizer (mocked)
# ---------------------------------------------------------------------------


def _make_mock_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


class TestClaudeSynthesizer:
    def test_returns_answer(self, tmp_path):
        png = tmp_path / "page1.png"
        png.write_bytes(b"\x89PNG\r\n")
        mock_response = _make_mock_response("The revenue was high [Page 1].")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            synth = ClaudeSynthesizer()
            result = synth.synthesize("what is revenue?", [(1, png)])
        assert isinstance(result, Answer)

    def test_model_stored(self):
        with patch("anthropic.Anthropic"):
            synth = ClaudeSynthesizer(model="claude-opus-4-8")
        assert synth.model == "claude-opus-4-8"

    def test_cited_pages_extracted(self, tmp_path):
        png2 = tmp_path / "page2.png"
        png2.write_bytes(b"\x89PNG\r\n")
        png5 = tmp_path / "page5.png"
        png5.write_bytes(b"\x89PNG\r\n")
        mock_response = _make_mock_response("See [Page 2] and [Page 5] for details.")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            synth = ClaudeSynthesizer()
            result = synth.synthesize("query", [(2, png2), (5, png5)])
        assert result.cited_pages == [2, 5]

    def test_duplicate_citations_deduplicated(self, tmp_path):
        png = tmp_path / "p.png"
        png.write_bytes(b"\x89PNG\r\n")
        mock_response = _make_mock_response("[Page 3] repeated [Page 3] again.")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            synth = ClaudeSynthesizer()
            result = synth.synthesize("q", [(3, png)])
        assert result.cited_pages == [3]

    def test_no_citations_returns_empty_list(self, tmp_path):
        png = tmp_path / "p.png"
        png.write_bytes(b"\x89PNG\r\n")
        mock_response = _make_mock_response("No specific pages mentioned.")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            synth = ClaudeSynthesizer()
            result = synth.synthesize("q", [(1, png)])
        assert result.cited_pages == []

    def test_answer_text_is_response_text(self, tmp_path):
        png = tmp_path / "p.png"
        png.write_bytes(b"\x89PNG\r\n")
        expected = "Profit was $1M [Page 1]."
        mock_response = _make_mock_response(expected)
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            synth = ClaudeSynthesizer()
            result = synth.synthesize("q", [(1, png)])
        assert result.text == expected

    def test_jpeg_uses_jpeg_media_type(self, tmp_path):
        jpg = tmp_path / "page.jpg"
        jpg.write_bytes(b"\xff\xd8\xff")
        mock_response = _make_mock_response("answer [Page 1].")
        with patch("anthropic.Anthropic") as MockClient:
            create_mock = MockClient.return_value.messages.create
            create_mock.return_value = mock_response
            synth = ClaudeSynthesizer()
            synth.synthesize("q", [(1, jpg)])
        call_args = create_mock.call_args
        content = call_args.kwargs["messages"][0]["content"]
        image_blocks = [b for b in content if b.get("type") == "image"]
        assert image_blocks[0]["source"]["media_type"] == "image/jpeg"

    def test_png_uses_png_media_type(self, tmp_path):
        png = tmp_path / "page.png"
        png.write_bytes(b"\x89PNG\r\n")
        mock_response = _make_mock_response("answer [Page 1].")
        with patch("anthropic.Anthropic") as MockClient:
            create_mock = MockClient.return_value.messages.create
            create_mock.return_value = mock_response
            synth = ClaudeSynthesizer()
            synth.synthesize("q", [(1, png)])
        call_args = create_mock.call_args
        content = call_args.kwargs["messages"][0]["content"]
        image_blocks = [b for b in content if b.get("type") == "image"]
        assert image_blocks[0]["source"]["media_type"] == "image/png"

    def test_import_error_without_anthropic(self):
        import sys
        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                ClaudeSynthesizer()

    def test_claude_synthesizer_satisfies_protocol(self):
        with patch("anthropic.Anthropic"):
            synth = ClaudeSynthesizer()
        assert isinstance(synth, Synthesizer)


# ---------------------------------------------------------------------------
# Module-level synthesize() convenience function
# ---------------------------------------------------------------------------


class TestSynthesizeFunction:
    def test_returns_answer(self, tmp_path):
        png = tmp_path / "p.png"
        png.write_bytes(b"\x89PNG\r\n")
        mock_response = _make_mock_response("answer [Page 1].")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            result = synthesize("question", [(1, png)])
        assert isinstance(result, Answer)

    def test_uses_specified_model(self, tmp_path):
        png = tmp_path / "p.png"
        png.write_bytes(b"\x89PNG\r\n")
        mock_response = _make_mock_response("answer.")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            result = synthesize("q", [(1, png)], model="claude-opus-4-8")
        assert result.model == "claude-opus-4-8"
