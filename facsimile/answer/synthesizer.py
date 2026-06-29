"""Vision-LLM synthesis over retrieved page images (stub for M3)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Answer:
    text: str
    cited_pages: list[int]
    model: str


def synthesize(
    question: str,
    page_images: list[tuple[int, Path]],
    model: str = "claude-opus-4-8",
) -> Answer:
    """
    Feed retrieved page images to a vision LLM and return a grounded answer.

    page_images: list of (page_number, image_path) in retrieval-score order.
    Implemented in M3.
    """
    raise NotImplementedError("Vision synthesis implemented in M3")
