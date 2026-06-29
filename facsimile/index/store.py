"""Qdrant multivector store for late-interaction page embeddings.

Each document page is stored as a variable-length list of patch vectors
(one per visual token / image patch).  Retrieval uses MaxSim — the same
late-interaction scoring used by ColBERT and ColPali.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

COLLECTION_NAME = "facsimile_pages"


@dataclass
class PageRecord:
    """One indexed document page."""

    doc_id: str
    page_number: int  # 1-based
    vectors: list[list[float]]  # shape [n_patches, patch_dim]
    payload: dict[str, Any] = field(default_factory=dict)


def make_client(url: str = "http://localhost:6333") -> QdrantClient:
    """Return a Qdrant client connected to *url*."""
    return QdrantClient(url=url)


def make_memory_client() -> QdrantClient:
    """Return an in-process Qdrant client backed by RAM (no server needed).

    Used in tests and local exploration.
    """
    return QdrantClient(":memory:")


def ensure_collection(client: QdrantClient, patch_dim: int) -> None:
    """Create the multivector collection if it does not already exist.

    *patch_dim* must match the embedding model's patch-vector dimension
    (e.g. 128 for ColPali-3B, 1536 for a larger variant).
    """
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "patches": VectorParams(
                size=patch_dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            )
        },
    )


def upsert_pages(client: QdrantClient, records: list[PageRecord]) -> None:
    """Insert or overwrite *records* in the collection.

    Point IDs are derived from ``{doc_id}:{page_number}`` so re-ingesting a
    document overwrites its previous vectors without leaving orphaned points.
    """
    points = [
        PointStruct(
            id=_page_id(r.doc_id, r.page_number),
            vector={"patches": r.vectors},
            payload={
                "doc_id": r.doc_id,
                "page_number": r.page_number,
                **r.payload,
            },
        )
        for r in records
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def _page_id(doc_id: str, page_number: int) -> int:
    """Deterministic integer point ID from doc_id + page_number.

    Uses a hash so IDs are stable across reingest.
    """
    return abs(hash(f"{doc_id}:{page_number}")) % (2**53)
