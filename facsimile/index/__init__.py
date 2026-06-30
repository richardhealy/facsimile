"""Index: multivector store backed by Qdrant.

Public surface:

- :class:`PageRecord` — one indexed page (doc coordinates + patch vectors).
- :func:`ensure_collection` — create the Qdrant collection if absent.
- :func:`upsert_pages` — write page records into the collection (idempotent).
- :func:`query_pages` — MaxSim search; returns top-k :class:`~qdrant_client.models.ScoredPoint` objects.
- :func:`get_client` — connect to a running Qdrant server.
- :func:`get_memory_client` — ephemeral in-memory client for tests.
"""

from facsimile.index.store import (
    PageRecord,
    ensure_collection,
    get_client,
    get_memory_client,
    query_pages,
    upsert_pages,
)

__all__ = [
    "PageRecord",
    "ensure_collection",
    "get_client",
    "get_memory_client",
    "upsert_pages",
    "query_pages",
]
