"""Vision-LLM synthesis over retrieved page images with citations (M3)."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Answer:
    """A synthesized answer grounded in retrieved page images.

    Attributes:
        text: Full answer text, which may reference pages as "Page N".
        cited_pages: Sorted list of page numbers explicitly cited in *text*.
        model: The vision LLM model identifier used.
    """

    text: str
    cited_pages: list[int]
    model: str


_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_PAGE_REF_RE = re.compile(r"\bpage\s+(\d+)", re.IGNORECASE)


def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for *path*."""
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return data, media_type


def _extract_cited_pages(text: str) -> list[int]:
    """Extract page numbers from "Page N" references in *text*."""
    return sorted(set(int(m) for m in _PAGE_REF_RE.findall(text)))


def synthesize(
    question: str,
    page_images: list[tuple[int, Path]],
    model: str = "claude-opus-4-8",
    api_key: str | None = None,
) -> Answer:
    """Feed retrieved page images to a vision LLM and return a grounded answer.

    Args:
        question: The natural-language question to answer.
        page_images: Ordered list of ``(page_num, image_path)`` in
            retrieval-score order.  Each image is base64-encoded and embedded
            directly in the prompt.
        model: Claude model to call.  Must support the vision capability.
        api_key: Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
            environment variable when *None*.

    Returns:
        :class:`Answer` with grounded text and extracted page citations.

    Raises:
        ImportError: When the ``anthropic`` package is not installed.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic is required for synthesize. "
            "Run: pip install facsimile[answer]"
        ) from exc

    client = anthropic.Anthropic(api_key=api_key)

    content: list[dict] = []
    for page_num, image_path in page_images:
        image_data, media_type = _encode_image(Path(image_path))
        content.append({"type": "text", "text": f"[Page {page_num}]"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                "Using the document pages shown above, answer the following question. "
                "When you reference information from a specific page, cite it as 'Page N' "
                "(e.g. 'Page 3').\n\n"
                f"Question: {question}"
            ),
        }
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )

    text = response.content[0].text
    return Answer(text=text, cited_pages=_extract_cited_pages(text), model=model)
