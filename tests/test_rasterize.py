"""Tests for facsimile.ingest.rasterize."""

import tempfile
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from facsimile.ingest.rasterize import (
    DEFAULT_DPI,
    page_count,
    rasterize_bytes,
    rasterize_pdf,
)


def _make_pdf(n_pages: int = 1) -> bytes:
    """Create a minimal PDF with *n_pages* blank pages."""
    import io
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(width=612, height=792)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# rasterize_bytes
# ---------------------------------------------------------------------------

def test_single_page_returns_one_image():
    pdf = _make_pdf(1)
    images = rasterize_bytes(pdf)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_multipage_returns_correct_count():
    pdf = _make_pdf(3)
    images = rasterize_bytes(pdf)
    assert len(images) == 3


def test_default_dpi_produces_reasonable_size():
    pdf = _make_pdf(1)
    images = rasterize_bytes(pdf, dpi=DEFAULT_DPI)
    # 612pt wide at 150 DPI => 612 * (150/72) ≈ 1275 px
    width, height = images[0].size
    expected_w = int(612 * DEFAULT_DPI / 72)
    expected_h = int(792 * DEFAULT_DPI / 72)
    assert abs(width - expected_w) <= 2
    assert abs(height - expected_h) <= 2


def test_higher_dpi_produces_larger_image():
    pdf = _make_pdf(1)
    lo = rasterize_bytes(pdf, dpi=72)[0]
    hi = rasterize_bytes(pdf, dpi=144)[0]
    assert hi.size[0] > lo.size[0]
    assert hi.size[1] > lo.size[1]


# ---------------------------------------------------------------------------
# rasterize_pdf (file path)
# ---------------------------------------------------------------------------

def test_rasterize_pdf_from_path():
    pdf = _make_pdf(2)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf)
        tmp_path = Path(f.name)
    try:
        images = rasterize_pdf(tmp_path)
        assert len(images) == 2
        assert all(isinstance(img, Image.Image) for img in images)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# page_count
# ---------------------------------------------------------------------------

def test_page_count():
    pdf = _make_pdf(4)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf)
        tmp_path = Path(f.name)
    try:
        assert page_count(tmp_path) == 4
    finally:
        tmp_path.unlink(missing_ok=True)
