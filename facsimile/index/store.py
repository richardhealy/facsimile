"""Qdrant multivector store for per-page ColPali/ColQwen embeddings.

Each document page is stored as a single Qdrant point whose vector is a
*list* of patch-level vectors (one per image patch produced by the visual
encoder). Qdrant's MaxSim comparator implements the late-interaction scoring
directly inside the ANN index, so retrieval is a single query call.

Payload schema per point:
  {
    "doc_id":   str,   # stable identifier for the source document
    "page_num": int,   # 1-based page number within the document
    "doc_path": str,   # original file path (for display)
  }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

COLLECTION_NAME = "facsimile_pages"
# ColPali / ColQwen2 project 128-dimensional patch vectors.
VECTOR_DIM = 128


@dataclass
class PageRecord:
    doc_id: str
    page_num: int
    doc_path: str
    vectors: list[list[float]]  # shape (n_patches, VECTOR_DIM)


def get_client(url: str = "http://localhost:6333") -> QdrantClient:
    """Return a QdrantClient pointed at *url*."""
    return QdrantClient(url=url)


def get_memory_client() -> QdrantClient:
    """Return an in-memory QdrantClient (no server required; useful for tests)."""
    return QdrantClient(":memory:")


def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    vector_dim: int = VECTOR_DIM,
) -> None:
    """Create the multivector collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM,
            ),
        ),
    )


def upsert_pages(
    client: QdrantClient,
    records: Sequence[PageRecord],
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Upsert page records into the collection.

    The point ID is derived deterministically from (doc_id, page_num) so that
    re-ingesting a document is idempotent.
    """
    points = [
        PointStruct(
            id=_page_id(r.doc_id, r.page_num),
            vector=r.vectors,
            payload={
                "doc_id": r.doc_id,
                "page_num": r.page_num,
                "doc_path": r.doc_path,
            },
        )
        for r in records
    ]
    client.upsert(collection_name=collection_name, points=points)


def query_pages(
    client: QdrantClient,
    query_vectors: list[list[float]],
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[ScoredPoint]:
    """Return the top-k pages scored by MaxSim against *query_vectors*."""
    return client.query_points(
        collection_name=collection_name,
        query=query_vectors,
        limit=top_k,
    ).points


def _page_id(doc_id: str, page_num: int) -> int:
    """Stable integer point ID from (doc_id, page_num) via a hash fold."""
    raw = hash((doc_id, page_num)) & 0xFFFF_FFFF_FFFF_FFFF
    # Qdrant unsigned 64-bit IDs; Python hash can be negative on 64-bit builds.
    return raw
