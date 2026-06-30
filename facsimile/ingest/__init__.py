"""Ingest: PDF → page images → late-interaction embeddings.

Public surface
--------------
:class:`~facsimile.ingest.embed.Embedder`
    Protocol that every embedder must satisfy (``embed_images``, ``embed_queries``).
:class:`~facsimile.ingest.embed.StubEmbedder`
    Deterministic random-vector embedder; no model download required.
:class:`~facsimile.ingest.embed.ColPaliEmbedder`
    ColPali v1.2 embedder (requires ``pip install facsimile[embed]``).
:class:`~facsimile.ingest.embed.ColQwenEmbedder`
    ColQwen2 embedder (requires ``pip install facsimile[embed]``).
:func:`~facsimile.ingest.rasterize.rasterize_pdf`
    PDF file path → list of PIL Images (one per page).
:func:`~facsimile.ingest.rasterize.rasterize_bytes`
    In-memory PDF bytes → list of PIL Images.
:func:`~facsimile.ingest.pipeline.ingest_pdf`
    Rasterize + embed without storing; returns :class:`~facsimile.index.store.PageRecord` list.
:func:`~facsimile.ingest.pipeline.ingest_and_store`
    Rasterize + embed + upsert into Qdrant; single-call pipeline.
"""

from facsimile.ingest.embed import ColPaliEmbedder, ColQwenEmbedder, Embedder, StubEmbedder
from facsimile.ingest.pipeline import ingest_and_store, ingest_pdf
from facsimile.ingest.rasterize import page_count, rasterize_bytes, rasterize_pdf

__all__ = [
    "Embedder",
    "StubEmbedder",
    "ColPaliEmbedder",
    "ColQwenEmbedder",
    "rasterize_pdf",
    "rasterize_bytes",
    "page_count",
    "ingest_pdf",
    "ingest_and_store",
]
