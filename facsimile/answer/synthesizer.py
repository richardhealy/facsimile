"""Vision-LLM synthesis over retrieved page images.

The public surface is the ``Synthesizer`` protocol.  Two implementations ship:

- ``StubSynthesizer`` — deterministic; no API call.  Used in tests and CI.
- ``ClaudeSynthesizer`` — sends page images to Claude via the Anthropic API.
  Requires ``pip install facsimile[answer]``.

Both accept a list of ``(page_number, image_path)`` tuples and return an
:class:`Answer` with the LLM response text and a list of cited page numbers.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class Answer:
    """Result from vision-LLM synthesis."""

    text: str
    cited_pages: list[int]
    model: str


@runtime_checkable
class Synthesizer(Protocol):
    """Anything that can answer a question over a list of page images."""

    def answer(
        self,
        question: str,
        page_images: list[tuple[int, Path]],
    ) -> Answer:
        """Return a grounded answer with page citations.

        Args:
            question: Natural-language question to answer.
            page_images: ``(page_number, image_path)`` pairs in retrieval-score
                order (highest score first).

        Returns:
            :class:`Answer` with the response text and cited page numbers.
        """
        ...


class StubSynthesizer:
    """Deterministic stub — no LLM call, suitable for tests and CI.

    Always cites every page that was provided and includes the question
    in the response text so tests can assert on content.
    """

    def __init__(self, model: str = "stub") -> None:
        self._model = model

    def answer(
        self,
        question: str,
        page_images: list[tuple[int, Path]],
    ) -> Answer:
        cited = sorted({pg for pg, _ in page_images})
        page_list = ", ".join(str(p) for p in cited) if cited else "none"
        text = f"[stub] Based on pages {page_list}: {question}"
        return Answer(text=text, cited_pages=cited, model=self._model)


_SYSTEM_PROMPT = (
    "You are a precise document analysis assistant. "
    "You will be shown one or more pages from a document, each labelled with its page number. "
    "Answer the user's question using only what is visible in the pages. "
    "Always cite the page numbers that support your answer, for example: "
    "'According to page 3, …' or 'See page 5.' "
    "If the answer spans multiple pages, cite all relevant ones."
)


def _media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/png")


def _image_to_b64(path: Path) -> tuple[str, str]:
    """Return ``(media_type, base64_data)`` for an image file."""
    return _media_type(path), base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _extract_cited_pages(text: str, available: set[int]) -> list[int]:
    """Extract page numbers mentioned in *text* that are in *available*."""
    found = {int(m) for m in re.findall(r"\bpage\s+(\d+)\b", text, re.IGNORECASE)}
    return sorted(found & available)


class ClaudeSynthesizer:
    """Send retrieved page images to Claude and return a grounded answer.

    Requires ``pip install facsimile[answer]`` (the ``anthropic`` package).

    Args:
        model: Claude model ID (default ``claude-opus-4-8``).
        max_tokens: Maximum tokens for the completion.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for ClaudeSynthesizer. "
                "Run: pip install facsimile[answer]"
            ) from exc
        self._client = _anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def answer(
        self,
        question: str,
        page_images: list[tuple[int, Path]],
    ) -> Answer:
        """Send *page_images* and *question* to Claude; extract page citations.

        Each image is prefixed with a ``[Page N]`` label so the model knows
        which page it is looking at.  Page numbers extracted from the response
        text that match the provided pages are returned in ``cited_pages``; if
        none are found the full set of provided pages is cited as a fallback.

        Args:
            question: Natural-language question to answer.
            page_images: ``(page_number, image_path)`` pairs in retrieval-score
                order (highest score first).

        Returns:
            :class:`Answer` with the response text and cited page numbers.
        """
        available = {pg for pg, _ in page_images}
        content: list[dict] = []
        for page_num, img_path in page_images:
            media_type, img_data = _image_to_b64(img_path)
            content.append({"type": "text", "text": f"[Page {page_num}]"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_data,
                },
            })
        content.append({"type": "text", "text": question})

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text
        cited = _extract_cited_pages(text, available) or sorted(available)
        return Answer(text=text, cited_pages=cited, model=self._model)


def synthesize(
    question: str,
    page_images: list[tuple[int, Path]],
    model: str = "claude-opus-4-8",
) -> Answer:
    """Convenience function: answer *question* using :class:`ClaudeSynthesizer`.

    Args:
        question: Natural-language question to answer.
        page_images: ``(page_number, image_path)`` pairs in retrieval-score order.
        model: Claude model ID to use.

    Returns:
        :class:`Answer` with response text and cited page numbers.
    """
    return ClaudeSynthesizer(model=model).answer(question, page_images)
