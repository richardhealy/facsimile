"""Qdrant-backed multivector page store with MaxSim late-interaction support."""

import hashlib
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

COLLECTION_NAME = "facsimile_pages"
VECTOR_DIM = 128


def _point_id(doc_id: str, page_number: int) -> str:
    """Stable UUID from doc_id + page_number so upserts are idempotent."""
    raw = f"{doc_id}:{page_number}".encode()
    return str(uuid.UUID(hashlib.md5(raw).hexdigest()))


class PageStore:
    """Stores and queries per-page multivector embeddings in Qdrant."""

    def __init__(self, url: str = "http://localhost:6333") -> None:
        self.client = QdrantClient(url=url)
        self.collection = COLLECTION_NAME

    def ensure_collection(self) -> None:
        """Create the collection with multivector config if it does not exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    def upsert_page(
        self,
        doc_id: str,
        page_number: int,
        vectors: list[list[float]],
        image_path: str | None = None,
    ) -> None:
        """Index one page as a multivector point; idempotent."""
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=_point_id(doc_id, page_number),
                    vector=vectors,
                    payload={
                        "doc_id": doc_id,
                        "page_number": page_number,
                        "image_path": image_path,
                    },
                )
            ],
        )

    def search(
        self, query_vectors: list[list[float]], top_k: int = 5
    ) -> list:
        """Return top-k pages by MaxSim late-interaction score."""
        return self.client.query_points(
            collection_name=self.collection,
            query=query_vectors,
            limit=top_k,
            with_payload=True,
        ).points

    def delete_document(self, doc_id: str) -> None:
        """Remove all pages for a document from the index."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
