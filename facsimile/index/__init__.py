"""Index: multivector store backed by Qdrant.

Each document page is a single Qdrant point whose vector field is a list of
patch vectors (one per ColPali/ColQwen image patch).  Qdrant's built-in
``MAX_SIM`` comparator implements late-interaction scoring directly inside the
ANN index, so retrieval is a single network round-trip.

Public surface
--------------
:class:`~facsimile.index.store.PageRecord`
    Dataclass holding a page's identity and its patch-vector matrix.
:func:`~facsimile.index.store.get_client`
    Connect to a running Qdrant server.
:func:`~facsimile.index.store.get_memory_client`
    In-memory Qdrant client (no server required; ideal for tests).
:func:`~facsimile.index.store.ensure_collection`
    Create the multivector collection if it does not already exist.
:func:`~facsimile.index.store.upsert_pages`
    Insert or overwrite :class:`~facsimile.index.store.PageRecord` objects.
:func:`~facsimile.index.store.query_pages`
    Return the top-k pages by MaxSim against a query-vector list.
"""

from facsimile.index.store import (
    COLLECTION_NAME,
    VECTOR_DIM,
    PageRecord,
    ensure_collection,
    get_client,
    get_memory_client,
    query_pages,
    upsert_pages,
)

__all__ = [
    "COLLECTION_NAME",
    "VECTOR_DIM",
    "PageRecord",
    "get_client",
    "get_memory_client",
    "ensure_collection",
    "upsert_pages",
    "query_pages",
]
