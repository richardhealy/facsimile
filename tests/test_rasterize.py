"""Tests for PDF rasterization."""

from pathlib import Path

import fitz
import pytest
from PIL import Image

from facsimile.ingest.rasterize import page_size_bytes, rasterize_pdf


def _make_pdf(path: Path, num_pages: int = 1) -> None:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((72, 100), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def test_single_page(tmp_path: Path) -> None:
    pdf = tmp_path / "single.pdf"
    _make_pdf(pdf, num_pages=1)
    images = rasterize_pdf(pdf)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].mode == "RGB"


def test_multi_page(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    _make_pdf(pdf, num_pages=5)
    images = rasterize_pdf(pdf)
    assert len(images) == 5


def test_dpi_scales_resolution(tmp_path: Path) -> None:
    pdf = tmp_path / "dpi.pdf"
    _make_pdf(pdf, num_pages=1)
    lo = rasterize_pdf(pdf, dpi=72)
    hi = rasterize_pdf(pdf, dpi=144)
    w_lo, h_lo = lo[0].size
    w_hi, h_hi = hi[0].size
    assert w_hi > w_lo
    assert h_hi > h_lo


def test_page_size_bytes() -> None:
    img = Image.new("RGB", (100, 200))
    assert page_size_bytes(img) == 100 * 200 * 3


def test_accepts_string_path(tmp_path: Path) -> None:
    pdf = tmp_path / "str.pdf"
    _make_pdf(pdf)
    images = rasterize_pdf(str(pdf))
    assert len(images) == 1
