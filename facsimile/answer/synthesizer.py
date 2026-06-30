"""Vision-LLM synthesis over retrieved page images.

The public entry point is :func:`synthesize`. It feeds up to *top_k* page
images to a Claude vision model and returns a grounded answer that cites
the page numbers the model drew from.

Page-citation extraction is light-weight: the model is prompted to emit
citations as ``[Page N]`` markers in its answer text, and we scan for them
with a regex.  The raw text is returned unchanged so the caller can render
or reformat it however they like.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class Answer:
    """A vision-grounded answer produced by :func:`synthesize`.

    Attributes:
        text: Raw model response text, including ``[Page N]`` citation markers.
        cited_pages: Sorted list of page numbers extracted from the response text.
        model: Exact model ID string echoed from the API response.
    """

    text: str
    cited_pages: list[int]
    model: str


_CITATION_RE = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are a document analysis assistant. "
    "You are shown one or more pages from a document (images). "
    "Answer the user's question using only the information visible in the pages. "
    "Cite every page you draw from using the marker [Page N], e.g. [Page 3]. "
    "If you cannot find the answer in the provided pages, say so clearly."
)


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def synthesize(
    question: str,
    page_images: list[tuple[int, Path]],
    model: str = "claude-opus-4-8",
    max_tokens: int = 1024,
    anthropic_client: Any = None,
) -> Answer:
    """Feed retrieved page images to a Claude vision model and return a grounded answer.

    Args:
        question: The natural-language question to answer.
        page_images: Ordered list of ``(page_number, image_path)`` pairs, highest
            relevance first.  Only paths to existing PNG/JPEG files are supported.
        model: Claude model ID to use.
        max_tokens: Maximum number of tokens in the response.
        anthropic_client: An ``anthropic.Anthropic`` instance.  If *None*, one
            is constructed from the ``ANTHROPIC_API_KEY`` environment variable.

    Returns:
        :class:`Answer` with the model's response text, the list of page numbers
        cited in the response, and the model ID used.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.
        ValueError: If *page_images* is empty.
    """
    if not page_images:
        raise ValueError("page_images must contain at least one entry")

    try:
        import anthropic as _anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for vision synthesis. "
            "Run: pip install anthropic"
        ) from exc

    client = anthropic_client or _anthropic.Anthropic()

    # Build the user message content: interleaved text labels + images.
    content: list[dict] = []
    for page_num, img_path in page_images:
        content.append(
            {"type": "text", "text": f"--- Page {page_num} ---"}
        )
        img = _load_image(Path(img_path))
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _image_to_base64(img),
                },
            }
        )

    content.append({"type": "text", "text": f"Question: {question}"})

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    text = response.content[0].text
    cited_pages = sorted(
        {int(m.group(1)) for m in _CITATION_RE.finditer(text)}
    )

    return Answer(text=text, cited_pages=cited_pages, model=response.model)
