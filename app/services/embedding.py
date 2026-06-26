from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL
from core.logger import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        model_name = EMBEDDING_MODEL   # default: "all-MiniLM-L6-v2"
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = _get_model()

    logger.info(f"Generating embeddings for {len(texts)} texts...")

    # encode() returns a numpy array of shape (N, embedding_dim)
    # convert_to_python_dtypes=False keeps numpy floats → faster
    embeddings = model.encode(
        texts,
        batch_size=32,          # process in batches to avoid memory spikes
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Convert numpy arrays → plain Python lists (JSON-serialisable)
    result = [embedding.tolist() for embedding in embeddings]

    logger.info(
        f"Generated {len(result)} embeddings "
        f"(dim={len(result[0])} each)."
    )

    return result


def embed_single(text: str) -> list[float]:
    
    if not text or not text.strip():
        raise ValueError("Cannot embed an empty string.")

    return embed_texts([text])[0]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    
    if not chunks:
        raise ValueError("Cannot embed an empty chunks list.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.info(f"Attached embeddings to {len(chunks)} chunks.")
    return chunks


def get_embedding_dimension() -> int:
    """Return the vector dimension of the current model (useful for DB setup)."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()