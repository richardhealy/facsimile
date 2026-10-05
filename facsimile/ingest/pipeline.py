"""End-to-end ingest pipeline: PDF → page images → embeddings → store."""

from __future__ import annotations

from pathlib import Path

from facsimile.index.store import PageRecord, delete_other_pages, upsert_pages
from facsimile.ingest.embed import Embedder
from facsimile.ingest.rasterize import rasterize_pdf

from qdrant_client import QdrantClient


def ingest_pdf(
    pdf_path: str | Path,
    embedder: Embedder,
    doc_id: str | None = None,
    dpi: int = 150,
) -> list[PageRecord]:
    """Rasterize *pdf_path* and embed each page; return the resulting records.

    The records are **not** stored — call :func:`store_records` or pass them
    directly to :func:`facsimile.index.store.upsert_pages`.

    Args:
        pdf_path: Path to the PDF file.
        embedder: Any :class:`facsimile.ingest.embed.Embedder` implementation.
        doc_id: Stable identifier for the document.  Defaults to the file stem.
        dpi: Rasterization resolution (higher = larger images, slower embedding).
    """
    path = Path(pdf_path)
    resolved_doc_id = doc_id or path.stem
    images = rasterize_pdf(path, dpi=dpi)
    vectors_per_page = embedder.embed_images(images)
    return [
        PageRecord(
            doc_id=resolved_doc_id,
            page_num=i + 1,
            doc_path=str(path),
            vectors=vecs,
        )
        for i, vecs in enumerate(vectors_per_page)
    ]


def ingest_and_store(
    pdf_path: str | Path,
    embedder: Embedder,
    client: QdrantClient,
    doc_id: str | None = None,
    dpi: int = 150,
    collection_name: str = "facsimile_pages",
) -> list[PageRecord]:
    """Ingest a PDF and store the resulting records in Qdrant.

    Re-ingesting a ``doc_id`` replaces it: the new pages are upserted first,
    then any other point stored for that ``doc_id`` (pages the new version
    lacks, or points from an older ID scheme) is deleted.

    Returns the list of :class:`~facsimile.index.store.PageRecord` objects so
    callers can inspect cost / patch counts without a second round-trip.
    """
    resolved_doc_id = doc_id or Path(pdf_path).stem
    records = ingest_pdf(pdf_path, embedder, doc_id=resolved_doc_id, dpi=dpi)
    upsert_pages(client, records, collection_name=collection_name)
    delete_other_pages(
        client,
        resolved_doc_id,
        [r.page_num for r in records],
        collection_name=collection_name,
    )
    return records
