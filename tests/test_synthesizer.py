"""Tests for M3 vision synthesis: Answer dataclass, synthesize(), helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from facsimile.answer.synthesizer import (
    Answer,
    _encode_image,
    _extract_cited_pages,
    synthesize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def white_png(tmp_path: Path) -> Path:
    """A minimal white 10×10 PNG on disk."""
    p = tmp_path / "page1.png"
    Image.new("RGB", (10, 10), color="white").save(p)
    return p


@pytest.fixture()
def two_pages(tmp_path: Path) -> list[tuple[int, Path]]:
    """Two page images as (page_num, path) pairs."""
    pages = []
    for i in range(1, 3):
        p = tmp_path / f"page{i}.png"
        Image.new("RGB", (10, 10), color=(i * 80, i * 80, i * 80)).save(p)
        pages.append((i, p))
    return pages


def _mock_anthropic(response_text: str):
    """Return a context-manager patch with a canned response text."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return patch("anthropic.Anthropic", return_value=mock_client)


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_fields(self):
        a = Answer(text="hello", cited_pages=[1, 2], model="claude-opus-4-8")
        assert a.text == "hello"
        assert a.cited_pages == [1, 2]
        assert a.model == "claude-opus-4-8"

    def test_empty_cited_pages(self):
        a = Answer(text="no pages", cited_pages=[], model="m")
        assert a.cited_pages == []


# ---------------------------------------------------------------------------
# _encode_image helper
# ---------------------------------------------------------------------------


class TestEncodeImage:
    def test_returns_tuple(self, white_png):
        data, media_type = _encode_image(white_png)
        assert isinstance(data, str)
        assert isinstance(media_type, str)

    def test_media_type_png(self, white_png):
        _, media_type = _encode_image(white_png)
        assert media_type == "image/png"

    def test_media_type_jpeg(self, tmp_path):
        p = tmp_path / "img.jpg"
        Image.new("RGB", (10, 10)).save(p)
        _, media_type = _encode_image(p)
        assert media_type == "image/jpeg"

    def test_media_type_unknown_defaults_to_png(self, tmp_path):
        p = tmp_path / "img.bmp"
        p.write_bytes(b"\x00" * 64)
        _, media_type = _encode_image(p)
        assert media_type == "image/png"

    def test_data_is_valid_base64(self, white_png):
        import base64

        data, _ = _encode_image(white_png)
        decoded = base64.standard_b64decode(data)
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# _extract_cited_pages helper
# ---------------------------------------------------------------------------


class TestExtractCitedPages:
    def test_single_reference(self):
        assert _extract_cited_pages("The answer is on Page 3.") == [3]

    def test_multiple_references(self):
        assert _extract_cited_pages("See Page 1 and Page 5.") == [1, 5]

    def test_duplicates_deduplicated(self):
        assert _extract_cited_pages("Page 2 confirms Page 2 data.") == [2]

    def test_case_insensitive(self):
        assert _extract_cited_pages("page 7 and PAGE 10") == [7, 10]

    def test_sorted_ascending(self):
        assert _extract_cited_pages("Page 5, Page 2, Page 9") == [2, 5, 9]

    def test_no_references_returns_empty(self):
        assert _extract_cited_pages("No citations here.") == []

    def test_partial_word_not_matched(self):
        # "homepage 3" should not match
        result = _extract_cited_pages("See homepage 3 for details.")
        assert result == []


# ---------------------------------------------------------------------------
# synthesize() — mocked Anthropic client
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_returns_answer_instance(self, two_pages):
        with _mock_anthropic("Revenue is $100M (Page 1)."):
            result = synthesize("What is revenue?", two_pages)
        assert isinstance(result, Answer)

    def test_text_matches_model_response(self, two_pages):
        with _mock_anthropic("Revenue is $100M (Page 1)."):
            result = synthesize("What is revenue?", two_pages)
        assert result.text == "Revenue is $100M (Page 1)."

    def test_cited_pages_extracted(self, two_pages):
        with _mock_anthropic("See Page 1 and Page 2 for details."):
            result = synthesize("Summarise.", two_pages)
        assert result.cited_pages == [1, 2]

    def test_model_field_set(self, two_pages):
        with _mock_anthropic("Answer on Page 1."):
            result = synthesize("Q?", two_pages, model="claude-haiku-4-5-20251001")
        assert result.model == "claude-haiku-4-5-20251001"

    def test_default_model(self, two_pages):
        with _mock_anthropic("Answer."):
            result = synthesize("Q?", two_pages)
        assert result.model == "claude-opus-4-8"

    def test_api_call_made_once(self, two_pages):
        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Answer.")]
            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client
            synthesize("Q?", two_pages)
        mock_client.messages.create.assert_called_once()

    def test_content_includes_image_blocks(self, two_pages):
        """Each page should produce a text label + an image block."""
        captured_content = []

        def capture_create(**kwargs):
            captured_content.extend(kwargs.get("messages", [{}])[0].get("content", []))
            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text="Page 1.")]
            return mock_resp

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = capture_create
            mock_cls.return_value = mock_client
            synthesize("Q?", two_pages)

        image_blocks = [b for b in captured_content if b.get("type") == "image"]
        assert len(image_blocks) == len(two_pages)

    def test_content_includes_page_labels(self, two_pages):
        captured_content = []

        def capture_create(**kwargs):
            captured_content.extend(kwargs.get("messages", [{}])[0].get("content", []))
            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text="ok")]
            return mock_resp

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = capture_create
            mock_cls.return_value = mock_client
            synthesize("Q?", two_pages)

        text_blocks = [b["text"] for b in captured_content if b.get("type") == "text"]
        assert any("[Page 1]" in t for t in text_blocks)
        assert any("[Page 2]" in t for t in text_blocks)

    def test_question_included_in_prompt(self, white_png):
        captured_content = []

        def capture_create(**kwargs):
            captured_content.extend(kwargs.get("messages", [{}])[0].get("content", []))
            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text="ok")]
            return mock_resp

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = capture_create
            mock_cls.return_value = mock_client
            synthesize("What is EBITDA?", [(1, white_png)])

        all_text = " ".join(b["text"] for b in captured_content if b.get("type") == "text")
        assert "What is EBITDA?" in all_text

    def test_empty_page_list(self, tmp_path):
        with _mock_anthropic("No pages given."):
            result = synthesize("Q?", [])
        assert isinstance(result, Answer)
        assert result.cited_pages == []

    def test_no_citations_in_response(self, white_png):
        with _mock_anthropic("The answer is 42."):
            result = synthesize("Q?", [(1, white_png)])
        assert result.cited_pages == []

    def test_api_key_passed_to_client(self, white_png):
        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="ok")]
            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client
            synthesize("Q?", [(1, white_png)], api_key="sk-test-key")
        mock_cls.assert_called_once_with(api_key="sk-test-key")
