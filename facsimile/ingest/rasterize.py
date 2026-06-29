"""Convert PDF files to per-page PIL Images for visual embedding."""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def rasterize_pdf(pdf_path: Path | str, dpi: int = 150) -> list[Image.Image]:
    """Render every page of *pdf_path* to a PIL Image at *dpi* dots-per-inch.

    Returns a list whose length equals the page count; index 0 is page 1.
    Higher DPI preserves more detail at the cost of larger tensors downstream.
    150 DPI is a good default — legible text, reasonable memory.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0  # pdfium renders at 72 DPI internally
    pages: list[Image.Image] = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale, rotation=0)
        pages.append(bitmap.to_pil())
    return pages


def rasterize_bytes(data: bytes, dpi: int = 150) -> list[Image.Image]:
    """Same as :func:`rasterize_pdf` but accepts raw PDF bytes.

    Useful when a PDF arrives over HTTP and has not been written to disk.
    """
    pdf = pdfium.PdfDocument(data)
    scale = dpi / 72.0
    pages: list[Image.Image] = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale, rotation=0)
        pages.append(bitmap.to_pil())
    return pages
