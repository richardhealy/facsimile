"""Ingest: PDF → page images → late-interaction embeddings.

Public surface:

- :class:`Embedder` — structural protocol; any object satisfying it works as an embedder.
- :class:`StubEmbedder` — deterministic random-vector embedder for tests and CI.
- :class:`ColPaliEmbedder` — ColPali / ColQwen2 (requires ``pip install facsimile[embed]``).
- :class:`ColQwenEmbedder` — ColQwen2 variant (same extra requirement).
- :func:`make_embedder`: build one of the three by name (``stub``, ``colpali``, ``colqwen``).
- :func:`rasterize_pdf` — PDF file path → list of PIL Images, one per page.
- :func:`rasterize_bytes` — PDF bytes → list of PIL Images (no temp file needed).
- :func:`page_count` — count pages without rasterizing.
- :func:`ingest_pdf` — rasterize + embed a PDF; return :class:`~facsimile.index.store.PageRecord` list.
- :func:`ingest_and_store` — full pipeline: PDF → embeddings → Qdrant.
"""

from facsimile.ingest.embed import (
    EMBEDDER_NAMES,
    ColPaliEmbedder,
    ColQwenEmbedder,
    Embedder,
    StubEmbedder,
    make_embedder,
)
from facsimile.ingest.pipeline import ingest_and_store, ingest_pdf
from facsimile.ingest.rasterize import page_count, rasterize_bytes, rasterize_pdf

__all__ = [
    "Embedder",
    "StubEmbedder",
    "ColPaliEmbedder",
    "ColQwenEmbedder",
    "EMBEDDER_NAMES",
    "make_embedder",
    "rasterize_pdf",
    "rasterize_bytes",
    "page_count",
    "ingest_pdf",
    "ingest_and_store",
]
