"""Qdrant-backed multivector store for page embeddings.

Each page is stored as a *list* of patch/token vectors (the ColPali output).
Qdrant's MAX_SIM comparator handles late-interaction scoring natively.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _page_uuid(doc_id: str, page: int) -> str:
    return str(uuid.uuid5(_NS, f"{doc_id}:{page}"))


class PageVectorStore:
    """Multivector store: one Qdrant point per document page.

    Each point's vector is a 2-D list ``(num_patches, vector_size)`` produced
    by the visual encoder.  Retrieval uses MAX_SIM late-interaction scoring,
    matching ColPali's ranking objective exactly.

    Args:
        location: Qdrant connection string.  Pass ``":memory:"`` for an
                  ephemeral in-process store (tests, demos).
        collection: Name of the Qdrant collection to use.
    """

    def __init__(
        self,
        location: str = "http://localhost:6333",
        collection: str = "facsimile_pages",
    ) -> None:
        self.client = QdrantClient(location=location)
        self.collection = collection

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def create_collection(self, vector_size: int = 128) -> None:
        """Create (or recreate) the multivector collection.

        Args:
            vector_size: Dimensionality of each patch vector.  ColPali uses
                         128; ColQwen2 uses 128 as well.
        """
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            ),
        )

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        doc_id: str,
        page: int,
        vectors: list[list[float]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index one page.

        Args:
            doc_id: Stable identifier for the source document (e.g. filename).
            page: Zero-based page index.
            vectors: Shape ``(num_patches, vector_size)`` — the ColPali output
                     for this page.
            metadata: Arbitrary payload stored alongside the vector.
        """
        payload: dict[str, Any] = {"doc_id": doc_id, "page": page}
        if metadata:
            payload.update(metadata)
        point = PointStruct(
            id=_page_uuid(doc_id, page),
            vector=vectors,
            payload=payload,
        )
        self.client.upsert(collection_name=self.collection, points=[point])

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_vectors: list[list[float]],
        top_k: int = 5,
    ) -> list[ScoredPoint]:
        """Retrieve the top-*k* pages by MAX_SIM late-interaction score.

        Args:
            query_vectors: Shape ``(num_query_tokens, vector_size)``.
            top_k: Number of pages to return.

        Returns:
            Qdrant ``ScoredPoint`` list sorted by descending score.
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vectors,
            limit=top_k,
        )
        return response.points

    def count(self) -> int:
        """Return the number of indexed pages."""
        return self.client.count(collection_name=self.collection).count
