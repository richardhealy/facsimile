"""Tests for PDF rasterization."""

from pathlib import Path

import fitz
import pytest
from PIL import Image

from facsimile.ingest.rasterize import pdf_page_count, rasterize_pdf


def _make_pdf(tmp_path: Path, num_pages: int = 3) -> Path:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1}", fontsize=24)
    p = tmp_path / "test.pdf"
    p.write_bytes(doc.tobytes())
    doc.close()
    return p


@pytest.fixture
def pdf3(tmp_path: Path) -> Path:
    return _make_pdf(tmp_path, num_pages=3)


def test_page_count(pdf3: Path) -> None:
    assert pdf_page_count(pdf3) == 3


def test_rasterize_yields_correct_count(pdf3: Path) -> None:
    pages = list(rasterize_pdf(pdf3))
    assert len(pages) == 3


def test_rasterize_page_numbers_one_based(pdf3: Path) -> None:
    nums = [n for n, _ in rasterize_pdf(pdf3)]
    assert nums == [1, 2, 3]


def test_rasterize_returns_pil_images(pdf3: Path) -> None:
    for _, img in rasterize_pdf(pdf3):
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"


def test_rasterize_image_dimensions_nonzero(pdf3: Path) -> None:
    for _, img in rasterize_pdf(pdf3, dpi=72):
        assert img.width > 0
        assert img.height > 0


def test_higher_dpi_gives_larger_image(pdf3: Path) -> None:
    lo = list(rasterize_pdf(pdf3, dpi=72))
    hi = list(rasterize_pdf(pdf3, dpi=144))
    _, img_lo = lo[0]
    _, img_hi = hi[0]
    assert img_hi.width > img_lo.width
    assert img_hi.height > img_lo.height


def test_single_page_pdf(tmp_path: Path) -> None:
    p = _make_pdf(tmp_path, num_pages=1)
    pages = list(rasterize_pdf(p))
    assert len(pages) == 1
    assert pages[0][0] == 1
