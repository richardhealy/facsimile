"""Tests for facsimile.ingest.embed and facsimile.ingest.pipeline."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from facsimile.ingest.embed import COLPALI_DIM, Embedder, StubEmbedder
from facsimile.ingest.pipeline import ingest_and_store, ingest_pdf
from facsimile.index.store import ensure_collection, get_memory_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_image(width: int = 612, height: int = 792) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _make_pdf_bytes(n_pages: int = 2) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(width=612, height=792)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_pdf_file(n_pages: int = 2) -> Path:
    data = _make_pdf_bytes(n_pages)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# StubEmbedder
# ---------------------------------------------------------------------------

class TestStubEmbedder:
    def test_implements_embedder_protocol(self):
        assert isinstance(StubEmbedder(), Embedder)

    def test_returns_one_result_per_image(self):
        emb = StubEmbedder()
        images = [_blank_image() for _ in range(3)]
        results = emb.embed_images(images)
        assert len(results) == 3

    def test_each_result_has_correct_patch_count(self):
        emb = StubEmbedder(n_patches=8)
        images = [_blank_image()]
        results = emb.embed_images(images)
        assert len(results[0]) == 8

    def test_each_patch_has_correct_dim(self):
        emb = StubEmbedder(dim=COLPALI_DIM)
        images = [_blank_image()]
        results = emb.embed_images(images)
        assert all(len(v) == COLPALI_DIM for v in results[0])

    def test_deterministic_with_same_seed(self):
        emb_a = StubEmbedder(seed=99)
        emb_b = StubEmbedder(seed=99)
        images = [_blank_image()]
        assert emb_a.embed_images(images) == emb_b.embed_images(images)

    def test_different_seeds_give_different_vectors(self):
        emb_a = StubEmbedder(seed=1)
        emb_b = StubEmbedder(seed=2)
        images = [_blank_image()]
        assert emb_a.embed_images(images) != emb_b.embed_images(images)

    def test_vectors_are_floats(self):
        emb = StubEmbedder(n_patches=4)
        images = [_blank_image()]
        patches = emb.embed_images(images)[0]
        assert all(isinstance(v, float) for v in patches[0])

    def test_empty_image_list(self):
        emb = StubEmbedder()
        assert emb.embed_images([]) == []


# ---------------------------------------------------------------------------
# ingest_pdf pipeline
# ---------------------------------------------------------------------------

class TestIngestPdf:
    def test_returns_one_record_per_page(self):
        path = _make_pdf_file(3)
        try:
            records = ingest_pdf(path, StubEmbedder())
            assert len(records) == 3
        finally:
            path.unlink(missing_ok=True)

    def test_page_numbers_are_one_based(self):
        path = _make_pdf_file(2)
        try:
            records = ingest_pdf(path, StubEmbedder())
            assert [r.page_num for r in records] == [1, 2]
        finally:
            path.unlink(missing_ok=True)

    def test_doc_id_defaults_to_file_stem(self):
        path = _make_pdf_file(1)
        try:
            records = ingest_pdf(path, StubEmbedder())
            assert records[0].doc_id == path.stem
        finally:
            path.unlink(missing_ok=True)

    def test_custom_doc_id(self):
        path = _make_pdf_file(1)
        try:
            records = ingest_pdf(path, StubEmbedder(), doc_id="annual_2023")
            assert all(r.doc_id == "annual_2023" for r in records)
        finally:
            path.unlink(missing_ok=True)

    def test_doc_path_stored(self):
        path = _make_pdf_file(1)
        try:
            records = ingest_pdf(path, StubEmbedder())
            assert records[0].doc_path == str(path)
        finally:
            path.unlink(missing_ok=True)

    def test_vectors_shape(self):
        path = _make_pdf_file(1)
        try:
            records = ingest_pdf(path, StubEmbedder(n_patches=16, dim=COLPALI_DIM))
            assert len(records[0].vectors) == 16
            assert len(records[0].vectors[0]) == COLPALI_DIM
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ingest_and_store (end-to-end round-trip)
# ---------------------------------------------------------------------------

class TestIngestAndStore:
    def test_pages_appear_in_qdrant(self):
        path = _make_pdf_file(3)
        client = get_memory_client()
        ensure_collection(client)
        try:
            records = ingest_and_store(path, StubEmbedder(), client)
            count = client.count(collection_name="facsimile_pages").count
            assert count == 3
        finally:
            path.unlink(missing_ok=True)

    def test_returns_page_records(self):
        path = _make_pdf_file(2)
        client = get_memory_client()
        ensure_collection(client)
        try:
            records = ingest_and_store(path, StubEmbedder(), client)
            assert len(records) == 2
            assert all(hasattr(r, "vectors") for r in records)
        finally:
            path.unlink(missing_ok=True)

    def test_re_ingest_is_idempotent(self):
        path = _make_pdf_file(2)
        client = get_memory_client()
        ensure_collection(client)
        try:
            ingest_and_store(path, StubEmbedder(), client)
            ingest_and_store(path, StubEmbedder(), client)
            count = client.count(collection_name="facsimile_pages").count
            assert count == 2
        finally:
            path.unlink(missing_ok=True)
