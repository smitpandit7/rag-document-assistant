import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq

from app.core import config
from app.core.logger import get_logger
from app.services.rag import answer_question, get_session_history, _append_to_history
from app.services.embedding import embed_single
from app.services.vector_store import query_chunks

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Schemas ────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:   str
    doc_id:     str | None = None
    session_id: str | None = None


class AskResponse(BaseModel):
    answer:     str
    sources:    list[dict]
    session_id: str
    doc_id:     str | None


# ── Standard Q&A ──────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    """
    Ask a question against uploaded document(s).

    - doc_id = None  → search across ALL documents
    - session_id = None → auto-generate a new session
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = body.session_id or str(uuid.uuid4())

    logger.info(f"[CHAT] session={session_id} | doc={body.doc_id or 'all'} | Q: {question[:80]}")

    try:
        result = answer_question(
            question=question,
            session_id=session_id,
            doc_id=body.doc_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[CHAT] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=session_id,
        doc_id=body.doc_id,
    )


# ── Streaming Q&A ─────────────────────────────────────────────────────────

@router.post("/ask/stream")
def ask_question_stream(body: AskRequest):
    """
    Streaming version of /ask.

    Returns tokens as Server-Sent Events (SSE) in real time.

    Event format:
        data: <token>        ← each token as it arrives
        data: [DONE]         ← signals end of stream

    Usage in curl:
        curl -X POST http://localhost:8000/chat/ask/stream
             -H "Content-Type: application/json"
             -d '{"question": "What is AI?", "doc_id": "abc"}'
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = body.session_id or str(uuid.uuid4())

    # ── Step 1: Retrieve relevant chunks (same as normal /ask) ────────────
    try:
        query_vector = embed_single(question)
        chunks = query_chunks(
            query_embedding=query_vector,
            doc_id=body.doc_id,
            top_k=5,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not chunks:
        def no_context_stream():
            msg = "I couldn't find relevant information in the uploaded documents."
            yield f"data: {msg}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_context_stream(), media_type="text/event-stream")

    # ── Step 2: Build prompt ───────────────────────────────────────────────
    context_block = "\n\n---\n\n".join([
        f"[Context {i}] (Source: {c.get('source_file', '')} p.{c.get('page_num', '')})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ])

    history = get_session_history(session_id)
    history_text = "\n".join([
        f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
        for h in history[-10:]
    ]) if history else ""

    prompt = f"""You are a helpful document assistant. Answer using ONLY the context below.
If the answer is not in the context, say so. Cite sources using [Context N].

{f'Conversation History:{chr(10)}{history_text}{chr(10)}' if history_text else ''}
Context:
{context_block}

Question: {question}
Answer:"""

    # ── Step 3: Stream from Groq ───────────────────────────────────────────
    def stream_tokens():
        client = Groq(api_key=config.GROQ_API_KEY)
        full_answer = []

        try:
            stream = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                stream=True,        # ← this is the key difference
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_answer.append(token)
                    # SSE format: "data: <token>\n\n"
                    yield f"data: {token}\n\n"

            # Signal end of stream
            yield "data: [DONE]\n\n"

            # Save full answer to chat history after streaming completes
            _append_to_history(session_id, "user", question)
            _append_to_history(session_id, "assistant", "".join(full_answer))

        except Exception as e:
            logger.error(f"[STREAM] Groq streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    logger.info(f"[STREAM] session={session_id} | doc={body.doc_id or 'all'} | Q: {question[:80]}")

    return StreamingResponse(
        stream_tokens(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",    # disables nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )