import uuid
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq

from app.core import config
from app.core.logger import get_logger
from app.core.auth import get_current_user
from app.services.rag import answer_question, get_session_history, _append_to_history
from app.services.embedding import embed_single
from app.services.vector_store import query_chunks

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


class AskRequest(BaseModel):
    question:   str
    doc_id:     str | None = None
    session_id: str | None = None


class AskResponse(BaseModel):
    answer:     str
    sources:    list[dict]
    session_id: str
    doc_id:     str | None


@router.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest, user: dict = Depends(get_current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    session_id = body.session_id or str(uuid.uuid4())
    try:
        result = answer_question(question=question, session_id=session_id, doc_id=body.doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[CHAT] error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    return AskResponse(answer=result["answer"], sources=result["sources"], session_id=session_id, doc_id=body.doc_id)


@router.post("/ask/stream")
def ask_question_stream(body: AskRequest, user: dict = Depends(get_current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    session_id = body.session_id or str(uuid.uuid4())

    try:
        query_vector = embed_single(question)
        chunks = query_chunks(query_embedding=query_vector, doc_id=body.doc_id, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not chunks:
        def no_context():
            yield "data: I couldn't find relevant information in the uploaded documents.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_context(), media_type="text/event-stream")

    context_block = "\n\n---\n\n".join([
        f"[Context {i}] (Source: {c.get('source_file','')} p.{c.get('page_num','')})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ])
    history = get_session_history(session_id)
    history_text = "\n".join([
        f"{'User' if h['role']=='user' else 'Assistant'}: {h['content']}"
        for h in history[-10:]
    ]) if history else ""

    prompt = f"""You are a helpful document assistant. Answer using ONLY the context below.
If the answer is not in the context, say so. Cite sources using [Context N].

{f'History:{chr(10)}{history_text}{chr(10)}' if history_text else ''}
Context:
{context_block}

Question: {question}
Answer:"""

    def stream_tokens():
        client = Groq(api_key=config.GROQ_API_KEY)
        full_answer = []
        try:
            stream = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_answer.append(token)
                    yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
            _append_to_history(session_id, "user", question)
            _append_to_history(session_id, "assistant", "".join(full_answer))
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})