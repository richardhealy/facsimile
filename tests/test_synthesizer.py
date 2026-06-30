"""Tests for M3 vision synthesis: synthesize() and Answer."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from facsimile.answer.synthesizer import Answer, _CITATION_RE, synthesize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(tmp_path: Path, name: str = "page.png", size=(64, 64)) -> Path:
    p = tmp_path / name
    img = Image.new("RGB", size, color=(200, 200, 200))
    img.save(p, format="PNG")
    return p


def _fake_client(text: str, model: str = "claude-opus-4-8") -> MagicMock:
    """Return a mock anthropic client whose .messages.create() returns *text*."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.model = model
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_fields(self):
        a = Answer(text="hello", cited_pages=[1, 3], model="claude-opus-4-8")
        assert a.text == "hello"
        assert a.cited_pages == [1, 3]
        assert a.model == "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Citation regex
# ---------------------------------------------------------------------------


class TestCitationRegex:
    def test_matches_page_n(self):
        m = _CITATION_RE.findall("See [Page 4] for details.")
        assert m == ["4"]

    def test_matches_multiple(self):
        m = _CITATION_RE.findall("[Page 1] and [Page 7] are relevant.")
        assert sorted(m) == ["1", "7"]

    def test_case_insensitive(self):
        m = _CITATION_RE.findall("[page 2]")
        assert m == ["2"]

    def test_no_match_without_brackets(self):
        m = _CITATION_RE.findall("Page 3 is relevant.")
        assert m == []

    def test_whitespace_variants(self):
        m = _CITATION_RE.findall("[Page  5]")
        assert m == ["5"]


# ---------------------------------------------------------------------------
# synthesize()
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_returns_answer(self, tmp_path):
        img = _make_png(tmp_path, "p1.png")
        client = _fake_client("Revenue is $10M [Page 1].")
        result = synthesize("What is revenue?", [(1, img)], anthropic_client=client)
        assert isinstance(result, Answer)

    def test_answer_text(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("The answer is 42 [Page 1].")
        result = synthesize("What is the answer?", [(1, img)], anthropic_client=client)
        assert result.text == "The answer is 42 [Page 1]."

    def test_cited_pages_extracted(self, tmp_path):
        img1 = _make_png(tmp_path, "p1.png")
        img2 = _make_png(tmp_path, "p2.png")
        client = _fake_client("See [Page 1] and [Page 2].")
        result = synthesize("question", [(1, img1), (2, img2)], anthropic_client=client)
        assert result.cited_pages == [1, 2]

    def test_cited_pages_sorted(self, tmp_path):
        img1 = _make_png(tmp_path, "p1.png")
        img2 = _make_png(tmp_path, "p3.png")
        client = _fake_client("Check [Page 3] then [Page 1].")
        result = synthesize("q", [(1, img1), (3, img2)], anthropic_client=client)
        assert result.cited_pages == [1, 3]

    def test_no_citations_gives_empty_list(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("I cannot find the answer in the pages.")
        result = synthesize("irrelevant", [(1, img)], anthropic_client=client)
        assert result.cited_pages == []

    def test_duplicate_citations_deduplicated(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("As noted [Page 1], and again [Page 1].")
        result = synthesize("q", [(1, img)], anthropic_client=client)
        assert result.cited_pages == [1]

    def test_model_returned(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("ok", model="claude-opus-4-8")
        result = synthesize("q", [(1, img)], anthropic_client=client, model="claude-opus-4-8")
        assert result.model == "claude-opus-4-8"

    def test_empty_page_images_raises(self):
        with pytest.raises(ValueError, match="page_images must contain at least one entry"):
            synthesize("q", [], anthropic_client=_fake_client("x"))

    def test_client_called_with_model_and_max_tokens(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("ok")
        synthesize(
            "q",
            [(1, img)],
            model="claude-opus-4-8",
            max_tokens=512,
            anthropic_client=client,
        )
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-8"
        assert call_kwargs["max_tokens"] == 512

    def test_system_prompt_included(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("ok")
        synthesize("q", [(1, img)], anthropic_client=client)
        call_kwargs = client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 10

    def test_message_contains_image_blocks(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("ok")
        synthesize("q", [(1, img)], anthropic_client=client)
        call_kwargs = client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        content_types = [block["type"] for block in messages[0]["content"]]
        assert "image" in content_types

    def test_page_label_in_message(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("ok")
        synthesize("q", [(5, img)], anthropic_client=client)
        call_kwargs = client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        text_blocks = [
            b["text"]
            for b in messages[0]["content"]
            if b["type"] == "text"
        ]
        assert any("Page 5" in t for t in text_blocks)

    def test_question_in_message(self, tmp_path):
        img = _make_png(tmp_path)
        client = _fake_client("ok")
        synthesize("What is the revenue growth?", [(1, img)], anthropic_client=client)
        call_kwargs = client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        text_blocks = [
            b["text"]
            for b in messages[0]["content"]
            if b["type"] == "text"
        ]
        assert any("What is the revenue growth?" in t for t in text_blocks)

    def test_multiple_pages_all_sent(self, tmp_path):
        imgs = [_make_png(tmp_path, f"p{i}.png") for i in range(3)]
        client = _fake_client("ok")
        synthesize("q", [(i + 1, p) for i, p in enumerate(imgs)], anthropic_client=client)
        call_kwargs = client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        image_blocks = [b for b in messages[0]["content"] if b["type"] == "image"]
        assert len(image_blocks) == 3

    def test_public_import_from_package(self):
        from facsimile.answer import Answer as A, synthesize as s
        assert A is Answer
        assert s is synthesize
