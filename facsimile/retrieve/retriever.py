"""Retrieval: query the Qdrant index and return scored pages."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from facsimile.index.store import COLLECTION_NAME, query_pages
from facsimile.ingest.embed import Embedder


@dataclass
class RetrievedPage:
    """A page returned by the retriever with its metadata and MaxSim score."""

    doc_id: str
    page_num: int
    doc_path: str
    score: float


def retrieve(
    query_vectors: list[list[float]],
    client: QdrantClient,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedPage]:
    """Return the top-k pages from the Qdrant index scored against *query_vectors*.

    Scoring is handled by Qdrant's built-in MaxSim comparator over the stored
    multivector (patch-level) embeddings — no secondary re-ranking needed.
    Results are sorted by score descending (most relevant first).

    Args:
        query_vectors: Query patch embeddings, shape ``(n_query_patches, dim)``.
        client: Connected :class:`~qdrant_client.QdrantClient`.
        top_k: Maximum number of pages to return.
        collection_name: Qdrant collection to query (default ``facsimile_pages``).
    """
    hits = query_pages(client, query_vectors, top_k=top_k, collection_name=collection_name)
    return [
        RetrievedPage(
            doc_id=hit.payload["doc_id"],
            page_num=hit.payload["page_num"],
            doc_path=hit.payload["doc_path"],
            score=hit.score,
        )
        for hit in hits
    ]


def retrieve_for_query(
    query_text: str,
    embedder: Embedder,
    client: QdrantClient,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedPage]:
    """Embed *query_text* then retrieve the top-k pages.

    This is the high-level entry point that combines query embedding with
    index lookup in one call.

    Args:
        query_text: Natural-language question or description to retrieve for.
        embedder: Any :class:`~facsimile.ingest.embed.Embedder` that implements
            ``embed_query``.
        client: Connected :class:`~qdrant_client.QdrantClient`.
        top_k: Maximum number of pages to return.
        collection_name: Qdrant collection to query.
    """
    query_vectors = embedder.embed_query(query_text)
    return retrieve(query_vectors, client, top_k=top_k, collection_name=collection_name)
