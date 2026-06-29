"""Rasterize PDF pages to PIL Images for downstream embedding."""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


DEFAULT_DPI = 150
_PDFIUM_BASE_DPI = 72


def rasterize_pdf(
    pdf_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> list[Image.Image]:
    """Return a PIL Image for every page in *pdf_path*, rendered at *dpi*."""
    scale = dpi / _PDFIUM_BASE_DPI
    doc = pdfium.PdfDocument(str(pdf_path))
    images: list[Image.Image] = []
    try:
        for page in doc:
            bitmap = page.render(scale=scale, rotation=0)
            images.append(bitmap.to_pil())
            page.close()
    finally:
        doc.close()
    return images


def rasterize_bytes(
    pdf_bytes: bytes,
    dpi: int = DEFAULT_DPI,
) -> list[Image.Image]:
    """Rasterize PDF bytes without writing to disk."""
    scale = dpi / _PDFIUM_BASE_DPI
    doc = pdfium.PdfDocument(pdf_bytes)
    images: list[Image.Image] = []
    try:
        for page in doc:
            bitmap = page.render(scale=scale, rotation=0)
            images.append(bitmap.to_pil())
            page.close()
    finally:
        doc.close()
    return images


def page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF without rasterizing."""
    doc = pdfium.PdfDocument(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
