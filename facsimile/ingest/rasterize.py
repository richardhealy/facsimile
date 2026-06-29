"""PDF rasterization: convert each page to a PIL Image."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def rasterize_pdf(pdf_path: Path | str, dpi: int = 150) -> list[Image.Image]:
    """Rasterize every page of *pdf_path* to a list of RGB PIL Images.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Render resolution. 150 dpi balances quality and embedding cost;
             use 72 for thumbnails or 300 for high-fidelity archiving.

    Returns:
        One PIL Image per page, in document order.
    """
    doc = fitz.open(str(pdf_path))
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pages: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(img)
    doc.close()
    return pages


def page_size_bytes(image: Image.Image) -> int:
    """Return the uncompressed byte size of a page image (width × height × 3)."""
    w, h = image.size
    return w * h * 3
