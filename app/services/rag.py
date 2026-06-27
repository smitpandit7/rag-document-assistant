"""
rag.py — Core RAG (Retrieval-Augmented Generation) Pipeline ⭐⭐⭐

Flow:
    User Question
        │
        ▼
    embed_single(question)          ← embedding.py
        │
        ▼
    query_chunks(embedding, doc_id) ← vector_store.py
        │
        ▼
    Build prompt with retrieved context + chat history
        │
        ▼
    Call Groq API (LLaMA 3.3 70B)
        │
        ▼
    Parse answer + extract source references
        │
        ▼
    Return { answer, sources, session_id }
"""

from groq import Groq
from app.core.config import GROQ_API_KEY, GROQ_MODEL
from app.core.logger import get_logger
from app.services.embedding import embed_single
from app.services.vector_store import query_chunks

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Groq client — initialised once at module load
# ---------------------------------------------------------------------------

_groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# In-memory chat history store
# Structure: { session_id: [ {"role": "user"|"assistant", "content": str}, ... ] }
# ---------------------------------------------------------------------------
_chat_history: dict[str, list[dict]] = {}

MAX_HISTORY_TURNS = 10   # keep last N turns to avoid context overflow


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    session_id: str,
    doc_id:   None,
    top_k: int = 5,
) -> dict:
    """
    Main RAG entry point. Given a question, retrieves relevant chunks
    and generates a grounded answer via Groq (LLaMA 3.3 70B).

    Args:
        question:   The user's natural-language question.
        session_id: Unique ID for this conversation (for chat history).
        doc_id:     If provided, restrict retrieval to this document.
                    If None, search across all documents.
        top_k:      Number of context chunks to retrieve.

    Returns:
        {
            "answer":     str,          # Groq's answer
            "sources":    list[dict],   # retrieved chunks used as context
            "session_id": str,
            "doc_id":     str | None,
        }
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    logger.info(f"[RAG] session={session_id} | doc={doc_id or 'all'} | Q: {question[:80]}")

    # ── Step 1: Embed the question ──────────────────────────────────────────
    query_vector = embed_single(question)

    # ── Step 2: Retrieve top-k relevant chunks ──────────────────────────────
    retrieved_chunks = query_chunks(
        query_embedding=query_vector,
        doc_id=doc_id,
        top_k=top_k,
    )

    if not retrieved_chunks:
        logger.warning("[RAG] No relevant chunks found.")
        answer = (
            "I couldn't find relevant information in the uploaded document(s) "
            "to answer your question. Please try rephrasing or upload a relevant document."
        )
        _append_to_history(session_id, "user", question)
        _append_to_history(session_id, "assistant", answer)
        return {
            "answer":     answer,
            "sources":    [],
            "session_id": session_id,
            "doc_id":     doc_id,
        }

    # ── Step 3: Build context block from retrieved chunks ───────────────────
    context_block = _build_context_block(retrieved_chunks)

    # ── Step 4: Get recent chat history for this session ────────────────────
    history = _get_history(session_id)
    history_block = _build_history_block(history)

    # ── Step 5: Build the final prompt ──────────────────────────────────────
    prompt = _build_prompt(
        question=question,
        context_block=context_block,
        history_block=history_block,
    )

    # ── Step 6: Call Groq ──────────────────────────────────────────────────
    logger.info("[RAG] Sending prompt to Groq...")
    try:
        response = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[RAG] Groq API error: {e}")
        raise RuntimeError(f"LLM generation failed: {e}")

    # ── Step 7: Persist to chat history ─────────────────────────────────────
    _append_to_history(session_id, "user", question)
    _append_to_history(session_id, "assistant", answer)

    # ── Step 8: Format source references ────────────────────────────────────
    sources = _format_sources(retrieved_chunks)

    logger.info(f"[RAG] Answer generated ({len(answer)} chars), {len(sources)} sources.")

    return {
        "answer":     answer,
        "sources":    sources,
        "session_id": session_id,
        "doc_id":     doc_id,
    }


def get_session_history(session_id: str) -> list[dict]:
    """
    Return the full chat history for a session.

    Returns:
        [ { "role": "user"|"assistant", "content": str }, ... ]
    """
    return _chat_history.get(session_id, [])


def clear_session_history(session_id: str) -> bool:
    """
    Delete all chat history for a session.
    Returns True if session existed, False otherwise.
    """
    if session_id in _chat_history:
        del _chat_history[session_id]
        logger.info(f"[RAG] Cleared history for session '{session_id}'.")
        return True
    return False


def list_sessions() -> list[str]:
    """Return all active session IDs."""
    return list(_chat_history.keys())


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

def _build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Each entry shows the source file and page number so the LLM can cite them.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source_label = _make_source_label(chunk)
        parts.append(
            f"[Context {i}] (Source: {source_label})\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_history_block(history: list[dict]) -> str:
    """Format recent chat turns into a readable history string."""
    if not history:
        return ""

    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:    # only last N turns
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")

    return "\n".join(lines)


def _build_prompt(
    question: str,
    context_block: str,
    history_block: str,
) -> str:
    """
    Compose the final prompt sent to Groq.

    Instructions:
    - Answer ONLY from the provided context.
    - If the answer is not in the context, say so explicitly.
    - Always cite sources using [Context N] references.
    - Be concise and factual.
    """
    history_section = ""
    if history_block:
        history_section = f"""
## Conversation History (for context only)
{history_block}

"""

    prompt = f"""You are a helpful document assistant. Answer the user's question using ONLY the information provided in the context sections below.

Rules:
1. Base your answer strictly on the provided context. Do NOT use outside knowledge.
2. If the answer is not present in the context, say: "I don't have enough information in the provided documents to answer this question."
3. Cite your sources inline using [Context N] references (e.g. "According to [Context 1], ...").
4. Be concise, accurate, and helpful.
5. If multiple contexts support the answer, reference all of them.
6. Do not mention "Context 1", "Context 2", or similar labels in your answer.
7. At the end of the answer, naturally mention the source file and page if appropriate.

{history_section}## Retrieved Context
{context_block}

## User Question
{question}

## Your Answer
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _get_history(session_id: str) -> list[dict]:
    return _chat_history.get(session_id, [])


def _append_to_history(session_id: str, role: str, content: str) -> None:
    if session_id not in _chat_history:
        _chat_history[session_id] = []
    _chat_history[session_id].append({"role": role, "content": content})

    # Trim to prevent unbounded growth (keep last MAX_HISTORY_TURNS * 2 messages)
    max_messages = MAX_HISTORY_TURNS * 2
    if len(_chat_history[session_id]) > max_messages:
        _chat_history[session_id] = _chat_history[session_id][-max_messages:]


# ---------------------------------------------------------------------------
# Source formatting helpers
# ---------------------------------------------------------------------------

def _format_sources(chunks: list[dict]) -> list[dict]:
    """
    Convert raw chunk dicts into clean source-reference objects for the API response.
    """
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        sources.append({
            "reference":   f"Context {i}",
            "chunk_id":    chunk.get("chunk_id", ""),
            "doc_id":      chunk.get("doc_id", ""),
            "source_file": chunk.get("source_file", ""),
            "page_num":    chunk.get("page_num", ""),
            "score":       chunk.get("score", 0.0),
            "text_preview": chunk["text"][:200] + "..."
                            if len(chunk["text"]) > 200
                            else chunk["text"],
        })
    return sources


def _make_source_label(chunk: dict) -> str:
    """Build a human-readable label like 'report.pdf, page 3'."""
    parts = []
    if chunk.get("source_file"):
        parts.append(chunk["source_file"])
    if chunk.get("page_num"):
        parts.append(f"page {chunk['page_num']}")
    return ", ".join(parts) if parts else chunk.get("doc_id", "unknown")