"""Tests for facsimile.ingest.rasterize."""

import io

import pypdfium2 as pdfium
import pytest
from PIL import Image

from facsimile.ingest.rasterize import rasterize_bytes, rasterize_pdf


def _make_pdf(n_pages: int = 1) -> bytes:
    """Return minimal valid PDF bytes with *n_pages* blank pages."""
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(width=595, height=842)  # A4 in points
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestRasterizeBytes:
    def test_single_page_returns_one_image(self):
        data = _make_pdf(1)
        pages = rasterize_bytes(data)
        assert len(pages) == 1

    def test_multipage_returns_correct_count(self):
        data = _make_pdf(3)
        pages = rasterize_bytes(data, dpi=72)
        assert len(pages) == 3

    def test_output_type_is_pil_image(self):
        data = _make_pdf(1)
        pages = rasterize_bytes(data)
        assert isinstance(pages[0], Image.Image)

    def test_dpi_affects_image_size(self):
        data = _make_pdf(1)
        lo = rasterize_bytes(data, dpi=72)[0]
        hi = rasterize_bytes(data, dpi=144)[0]
        assert hi.width > lo.width
        assert hi.height > lo.height


class TestRasterizePdf:
    def test_rasterize_pdf_from_path(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(_make_pdf(2))
        pages = rasterize_pdf(pdf_path)
        assert len(pages) == 2
        assert all(isinstance(p, Image.Image) for p in pages)
