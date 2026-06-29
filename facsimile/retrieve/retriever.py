"""Retrieval layer: text query → late-interaction page ranking.

Combines a text query encoder with the Qdrant MaxSim index to return
the top-k most visually relevant pages for a natural-language question.

Public surface:
  TextQueryEmbedder — protocol satisfied by both Stub and ColPali implementations.
  StubTextEmbedder  — deterministic random vectors, no model required (tests/CI).
  ColPaliTextEmbedder — ColPali text encoder (requires ``pip install facsimile[embed]``).
  Retriever         — query → RetrievalResult list, backed by Qdrant MaxSim.
  RetrievalResult   — dataclass returned per matched page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from qdrant_client import QdrantClient

from facsimile.index.store import COLLECTION_NAME, query_pages
from facsimile.ingest.embed import COLPALI_DIM


@runtime_checkable
class TextQueryEmbedder(Protocol):
    """Anything that can encode a text query into patch-compatible vectors.

    Returns shape ``(n_tokens, dim)`` — the same ``dim`` as the page patch
    vectors so that MaxSim scoring works across modalities.
    """

    def embed_query(self, text: str) -> list[list[float]]:
        """Embed *text* and return a list of token vectors."""
        ...


@dataclass
class RetrievalResult:
    """A single retrieved page with its provenance and relevance score."""

    doc_id: str
    page_num: int
    doc_path: str
    score: float


class StubTextEmbedder:
    """Deterministic random-vector text embedder — no model required.

    Seeds the RNG with both the constructor seed and a hash of the query
    text so the same question always produces the same vectors while
    different questions produce different vectors.
    """

    def __init__(
        self,
        n_tokens: int = 8,
        dim: int = COLPALI_DIM,
        seed: int = 42,
    ) -> None:
        self._n_tokens = n_tokens
        self._dim = dim
        self._seed = seed

    def embed_query(self, text: str) -> list[list[float]]:
        text_hash = hash(text) % (2**32)
        rng = np.random.default_rng(self._seed ^ text_hash)
        return rng.standard_normal((self._n_tokens, self._dim)).tolist()


class ColPaliTextEmbedder:
    """ColPali text query encoder backed by ``colpali-engine``.

    Requires ``pip install facsimile[embed]``.  Shares the same model
    checkpoint as :class:`facsimile.ingest.embed.ColPaliEmbedder` —
    load one instance per process and reuse it.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as exc:
            raise ImportError(
                "colpali-engine and torch are required for ColPaliTextEmbedder. "
                "Run: pip install facsimile[embed]"
            ) from exc

        self._torch = torch
        self._device = device
        self._processor = ColPaliProcessor.from_pretrained(model_name)
        self._model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

    def embed_query(self, text: str) -> list[list[float]]:
        batch = self._processor.process_queries([text]).to(self._device)
        with self._torch.no_grad():
            embeddings = self._model(**batch)  # (1, n_tokens, dim)
        return embeddings[0].float().cpu().tolist()


class Retriever:
    """Retrieve the most visually relevant pages for a text query.

    Embeds the query with *embedder*, then issues a single Qdrant query
    that scores all indexed pages by MaxSim and returns the top-k.

    Example::

        client = get_memory_client()
        ensure_collection(client)
        ingest_and_store("report.pdf", ColPaliEmbedder(), client)

        retriever = Retriever(client, ColPaliTextEmbedder())
        results = retriever.retrieve("What was the operating margin in Q3?", top_k=3)
        for r in results:
            print(r.page_num, r.score)
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: TextQueryEmbedder,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Embed *query* and return the top-*top_k* pages by MaxSim score."""
        query_vectors = self._embedder.embed_query(query)
        points = query_pages(
            self._client,
            query_vectors=query_vectors,
            top_k=top_k,
            collection_name=self._collection_name,
        )
        return [
            RetrievalResult(
                doc_id=p.payload["doc_id"],
                page_num=p.payload["page_num"],
                doc_path=p.payload["doc_path"],
                score=p.score,
            )
            for p in points
        ]
