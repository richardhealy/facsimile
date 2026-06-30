"""Vision-LLM synthesis over retrieved page images with page citations."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class Answer:
    text: str
    cited_pages: list[int]
    model: str


@runtime_checkable
class Synthesizer(Protocol):
    """Anything that can answer a question given page images."""

    def synthesize(self, question: str, page_images: list[tuple[int, Path]]) -> Answer:
        """Return an answer grounded in the supplied page images.

        Args:
            question: Natural-language question.
            page_images: Ordered list of ``(page_number, image_path)`` pairs.

        Returns:
            :class:`Answer` with extracted citation page numbers.
        """
        ...


class StubSynthesizer:
    """Deterministic synthesizer — no model required.

    Returns a canned response citing every supplied page.  Useful for tests
    and CI where making real API calls is not feasible.
    """

    def __init__(self, model: str = "stub") -> None:
        self.model = model

    def synthesize(self, question: str, page_images: list[tuple[int, Path]]) -> Answer:
        pages = [num for num, _ in page_images]
        cited = " ".join(f"[Page {n}]" for n in pages) if pages else ""
        text = f"[STUB] Answer to '{question}'. {cited}".strip()
        return Answer(text=text, cited_pages=pages, model=self.model)


class ClaudeSynthesizer:
    """Vision-LLM synthesizer backed by Claude via the Anthropic SDK.

    Requires ``pip install facsimile[answer]``.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for ClaudeSynthesizer. "
                "Run: pip install facsimile[answer]"
            ) from exc
        self._client = anthropic.Anthropic()
        self.model = model
        self._max_tokens = max_tokens

    def synthesize(self, question: str, page_images: list[tuple[int, Path]]) -> Answer:
        content: list[dict] = []
        for page_num, image_path in page_images:
            suffix = Path(image_path).suffix.lower()
            media_type = "image/png" if suffix == ".png" else "image/jpeg"
            with open(image_path, "rb") as fh:
                data = base64.standard_b64encode(fh.read()).decode()
            content.append({"type": "text", "text": f"[Page {page_num}]"})
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    f"Question: {question}\n\n"
                    "Answer based on the page images above. "
                    "Cite page numbers using [Page N] notation."
                ),
            }
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        answer_text = response.content[0].text
        cited_pages = sorted(
            {int(m) for m in re.findall(r"\[Page (\d+)\]", answer_text)}
        )
        return Answer(text=answer_text, cited_pages=cited_pages, model=self.model)


def synthesize(
    question: str,
    page_images: list[tuple[int, Path]],
    model: str = "claude-opus-4-8",
) -> Answer:
    """Convenience wrapper — creates a :class:`ClaudeSynthesizer` and calls it."""
    return ClaudeSynthesizer(model=model).synthesize(question, page_images)
