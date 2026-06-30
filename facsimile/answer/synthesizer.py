"""Vision-LLM synthesis over retrieved page images with page citations.

Public surface:

- ``Answer`` — result dataclass returned by all synthesizers.
- ``Synthesizer`` — structural protocol; any object with a ``synthesize``
  method that matches the signature satisfies it.
- ``StubSynthesizer`` — deterministic, no model or API key required.  Used
  in tests and CI.
- ``ClaudeSynthesizer`` — sends page images to the Anthropic vision API and
  extracts cited page numbers from the response.  Requires
  ``pip install facsimile[vision]``.
- ``synthesize()`` — one-call convenience wrapper around
  ``ClaudeSynthesizer``.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class Answer:
    """Synthesis result returned by a :class:`Synthesizer`."""

    text: str
    cited_pages: list[int]
    model: str


@runtime_checkable
class Synthesizer(Protocol):
    """Anything that can answer a question from a ranked list of page images."""

    def synthesize(
        self,
        question: str,
        page_images: list[tuple[int, Path]],
    ) -> Answer:
        """Return a grounded answer with cited page numbers.

        Args:
            question: Natural-language question to answer.
            page_images: Ordered list of ``(page_number, image_path)`` pairs
                in retrieval-score order (highest first).

        Returns:
            An :class:`Answer` whose ``cited_pages`` lists the page numbers
            referenced in the response text.
        """
        ...


class StubSynthesizer:
    """Deterministic stub — no API key or model download required.

    Returns a canned answer that mentions every supplied page number.
    Useful in tests and CI where a live vision LLM is unavailable.
    """

    def __init__(self, model: str = "stub") -> None:
        self._model = model

    def synthesize(
        self,
        question: str,
        page_images: list[tuple[int, Path]],
    ) -> Answer:
        pages = [p for p, _ in page_images]
        cited_text = ", ".join(f"page {p}" for p in pages) if pages else "no pages"
        return Answer(
            text=f"Stub answer for: {question}. Evidence found on {cited_text}.",
            cited_pages=pages,
            model=self._model,
        )


_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ClaudeSynthesizer:
    """Vision-LLM synthesis using the Anthropic Claude API.

    Page images are encoded as base64 and sent in a single multi-image
    message.  The model is instructed to cite relevant pages using the
    format ``(page N)``, and those references are parsed from the response
    text to populate :attr:`Answer.cited_pages`.

    Requires ``pip install facsimile[vision]``.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for ClaudeSynthesizer. "
                "Run: pip install facsimile[vision]"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def synthesize(
        self,
        question: str,
        page_images: list[tuple[int, Path]],
    ) -> Answer:
        content: list[dict] = []

        for page_num, img_path in page_images:
            img_bytes = Path(img_path).read_bytes()
            b64 = base64.standard_b64encode(img_bytes).decode()
            media_type = _MEDIA_TYPES.get(Path(img_path).suffix.lower(), "image/png")

            content.append({"type": "text", "text": f"[Page {page_num}]"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                }
            )

        content.append(
            {
                "type": "text",
                "text": (
                    f"Question: {question}\n\n"
                    "Answer using only the page images provided above. "
                    "Cite every page that supports your answer using the format (page N). "
                    "If the answer is spread across multiple pages, cite all relevant pages."
                ),
            }
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": content}],
        )

        text: str = response.content[0].text
        cited = sorted(
            {int(m) for m in re.findall(r"\bpage\s+(\d+)", text, re.IGNORECASE)}
        )

        return Answer(text=text, cited_pages=cited, model=self._model)


def synthesize(
    question: str,
    page_images: list[tuple[int, Path]],
    model: str = "claude-opus-4-8",
    api_key: str | None = None,
) -> Answer:
    """Convenience wrapper that creates a :class:`ClaudeSynthesizer` and calls it.

    Args:
        question: Natural-language question to answer.
        page_images: ``(page_number, image_path)`` pairs in score order.
        model: Anthropic model identifier.
        api_key: Anthropic API key; falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns:
        :class:`Answer` with response text and cited page numbers.
    """
    return ClaudeSynthesizer(model=model, api_key=api_key).synthesize(
        question, page_images
    )
