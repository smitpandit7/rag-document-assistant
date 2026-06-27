from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger
from app.services.rag import get_session_history, clear_session_history, list_sessions

logger = get_logger(__name__)
router = APIRouter(prefix="/history", tags=["Chat History"])


@router.get("/{session_id}")
def get_history(session_id: str):
    """
    Retrieve full chat history for a session.

    Returns all turns in order:
    [
        { "role": "user",      "content": "What is AI?" },
        { "role": "assistant", "content": "AI is ..."   },
        ...
    ]
    """
    history = get_session_history(session_id)

    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for session '{session_id}'."
        )

    return {
        "session_id":  session_id,
        "total_turns": len(history),
        "history":     history,
    }


@router.delete("/{session_id}")
def clear_history(session_id: str):
    """
    Clear all chat history for a session.
    """
    deleted = clear_session_history(session_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for session '{session_id}'."
        )

    return {
        "message":    f"Chat history for session '{session_id}' cleared.",
        "session_id": session_id,
    }


@router.get("/")
def get_all_sessions():
    """
    List all active session IDs.
    """
    sessions = list_sessions()
    return {
        "total_sessions": len(sessions),
        "sessions":       sessions,
    }