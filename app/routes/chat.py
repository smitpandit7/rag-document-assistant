import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logger import get_logger
from app.services.rag import answer_question

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Request / Response schemas ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question:   str
    doc_id:     str | None = None    # None = search across all documents
    session_id: str | None = None    # None = auto-generate a new session


class AskResponse(BaseModel):
    answer:     str
    sources:    list[dict]
    session_id: str
    doc_id:     str | None


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    """
    Ask a question against uploaded document(s).

    - If doc_id is provided → search only that document.
    - If doc_id is None     → search across ALL uploaded documents.
    - If session_id is None → a new session is created automatically.

    Returns the answer, source references, and session_id
    (use this session_id in follow-up questions for chat history).

    Example request:
    {
        "question":   "What are the main challenges of AI?",
        "doc_id":     "abc123_report",
        "session_id": "user_session_001"
    }
    """
    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Auto-generate session_id if not provided
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(
        f"[CHAT] session={session_id} | doc={body.doc_id or 'all'} | Q: {question[:80]}"
    )

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