from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.core import config
from app.core.logger import get_logger
from app.routes import upload, chat, history

logger = get_logger(__name__)

app = FastAPI(
    title="AI Document Assistant",
    description="RAG-powered API for answering questions from PDF documents.",
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(history.router)


# ── Fix Swagger UI for multi-file upload ───────────────────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Fix the upload endpoint so Swagger shows a proper multi-file picker
    try:
        upload_schema = schema["paths"]["/documents/upload"]["post"]
        upload_schema["requestBody"]["content"]["multipart/form-data"]["schema"] = {
            "type": "object",
            "properties": {
                "files": {
                    "type":  "array",
                    "items": {"type": "string", "format": "binary"},
                }
            },
            "required": ["files"],
        }
    except KeyError:
        pass

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Starting AI Document Assistant...")

    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    from pathlib import Path
    for d in [config.UPLOAD_DIR, config.CHROMA_DB_PATH, config.LOG_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("All systems ready.")


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "ok",
        "app":     "AI Document Assistant",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":    "healthy",
        "model":     config.GROQ_MODEL,
        "embedding": config.EMBEDDING_MODEL,
    }