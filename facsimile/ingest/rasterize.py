"""PDF → per-page PIL images via PyMuPDF (no external OCR dependency)."""

from pathlib import Path
from typing import Generator

import fitz  # pymupdf
from PIL import Image


def rasterize_pdf(
    pdf_path: Path | str, dpi: int = 150
) -> Generator[tuple[int, Image.Image], None, None]:
    """Yield (page_number, image) for every page; page_number is 1-based."""
    pdf_path = Path(pdf_path)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    doc = fitz.open(pdf_path)
    try:
        for idx in range(len(doc)):
            pix = doc[idx].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yield idx + 1, img
    finally:
        doc.close()


def pdf_page_count(pdf_path: Path | str) -> int:
    """Return the number of pages in a PDF without rasterizing."""
    doc = fitz.open(Path(pdf_path))
    n = len(doc)
    doc.close()
    return n
