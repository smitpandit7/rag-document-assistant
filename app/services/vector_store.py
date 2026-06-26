import chromadb
from chromadb.config import Settings as ChromaSettings
from core.config import CHROMA_DB_PATH
from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ChromaDB client — singleton, persistent on disk at settings.CHROMA_DB_PATH
# ---------------------------------------------------------------------------

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Lazy-init a persistent ChromaDB client."""
    global _client
    if _client is None:
        logger.info(f"Initialising ChromaDB at: {CHROMA_DB_PATH}")
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client ready.")
    return _client


def _get_collection(doc_id: str) -> chromadb.Collection:
   
    client = _get_client()

    # Sanitise doc_id: ChromaDB collection names must be 3-63 chars,
    # alphanumeric + hyphens/underscores, no leading/trailing hyphens.
    safe_name = _sanitise_collection_name(doc_id)

    collection = client.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
    )
    return collection


def _get_shared_collection() -> chromadb.Collection:
    """Shared collection that holds chunks from ALL documents (multi-doc search)."""
    client = _get_client()
    return client.get_or_create_collection(
        name="all_documents",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict]) -> int:
    
    if not chunks:
        raise ValueError("No chunks provided to store.")

    # Validate that embeddings are present
    for chunk in chunks:
        if "embedding" not in chunk:
            raise ValueError(
                f"Chunk '{chunk.get('chunk_id')}' is missing its embedding. "
                "Run embed_chunks() before store_chunks()."
            )

    doc_id = chunks[0]["doc_id"]

    ids         = [c["chunk_id"]  for c in chunks]
    documents   = [c["text"]      for c in chunks]
    embeddings  = [c["embedding"] for c in chunks]
    metadatas   = [
        {
            "doc_id":      c["doc_id"],
            "source_file": c.get("source_file", ""),
            "page_num":    str(c.get("page_num", "")),    # Chroma metadata = strings
            "chunk_index": str(c.get("chunk_index", "")),
            "char_start":  str(c.get("char_start", "")),
            "char_end":    str(c.get("char_end", "")),
        }
        for c in chunks
    ]

    # 1. Store in per-document collection
    per_doc_col = _get_collection(doc_id)
    per_doc_col.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # 2. Store in shared collection (for multi-doc queries)
    shared_col = _get_shared_collection()
    shared_col.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks for doc '{doc_id}'.")
    return len(chunks)


def query_chunks(
    query_embedding: list[float],
    doc_id: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    
    if doc_id:
        collection = _get_collection(doc_id)
    else:
        collection = _get_shared_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),   # can't ask for more than exist
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns lists-of-lists (one per query); we sent one query so take [0]
    raw_ids       = results["ids"][0]
    raw_docs      = results["documents"][0]
    raw_metas     = results["metadatas"][0]
    raw_distances = results["distances"][0]

    hits = []
    for chunk_id, text, meta, distance in zip(
        raw_ids, raw_docs, raw_metas, raw_distances
    ):
        # ChromaDB cosine distance = 1 - similarity → convert back to similarity
        similarity = round(1 - distance, 4)

        hits.append({
            "chunk_id":    chunk_id,
            "text":        text,
            "score":       similarity,
            "doc_id":      meta.get("doc_id", ""),
            "source_file": meta.get("source_file", ""),
            "page_num":    meta.get("page_num", ""),
            "chunk_index": meta.get("chunk_index", ""),
        })

    logger.info(
        f"Query returned {len(hits)} chunks "
        f"(doc_id={doc_id or 'all'}, top_k={top_k})"
    )

    return hits


def delete_document(doc_id: str) -> bool:
   
    client = _get_client()
    safe_name = _sanitise_collection_name(doc_id)

    deleted = False

    # Delete per-document collection entirely
    try:
        client.delete_collection(safe_name)
        logger.info(f"Deleted collection '{safe_name}'.")
        deleted = True
    except Exception:
        logger.warning(f"Collection '{safe_name}' not found, skipping.")

    # Remove this doc's chunks from shared collection by metadata filter
    try:
        shared_col = _get_shared_collection()
        shared_col.delete(where={"doc_id": doc_id})
        logger.info(f"Removed doc '{doc_id}' chunks from shared collection.")
    except Exception as e:
        logger.warning(f"Could not remove from shared collection: {e}")

    return deleted


def list_documents() -> list[str]:
    
    client = _get_client()
    collections = client.list_collections()
    return [
        col.name
        for col in collections
        if col.name != "all_documents"
    ]


def get_document_chunk_count(doc_id: str) -> int:
    """Return the number of chunks stored for a given document."""
    col = _get_collection(doc_id)
    return col.count()



def _sanitise_collection_name(name: str) -> str:
    """
    ChromaDB collection names:
    - 3–63 characters
    - alphanumeric, hyphens, underscores only
    - cannot start or end with a hyphen
    """
    import re
    sanitised = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitised = sanitised.strip("-")
    sanitised = sanitised[:63]
    if len(sanitised) < 3:
        sanitised = sanitised.ljust(3, "_")
    return sanitised