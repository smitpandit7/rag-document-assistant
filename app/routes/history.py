from fastapi import APIRouter, HTTPException, Depends
from app.core.auth import get_current_user
from app.core.logger import get_logger
from app.services.rag import get_session_history, clear_session_history, list_sessions

logger = get_logger(__name__)
router = APIRouter(prefix="/history", tags=["Chat History"])


@router.get("/")
def get_all_sessions(user: dict = Depends(get_current_user)):
    sessions = list_sessions()
    return {"total_sessions": len(sessions), "sessions": sessions}


@router.get("/{session_id}")
def get_history(session_id: str, user: dict = Depends(get_current_user)):
    history = get_session_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No history for session '{session_id}'.")
    return {"session_id": session_id, "total_turns": len(history), "history": history}


@router.delete("/{session_id}")
def clear_history(session_id: str, user: dict = Depends(get_current_user)):
    deleted = clear_session_history(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No history for session '{session_id}'.")
    return {"message": f"History for session '{session_id}' cleared."}