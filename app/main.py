from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.core import config
from app.core.logger import get_logger
from app.core.database import init_db
from app.routes import upload, chat, history, auth

logger = get_logger(__name__)

app = FastAPI(
    title="AI Document Assistant",
    description="RAG-powered API for answering questions from PDF documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(history.router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    try:
        upload_schema = schema["paths"]["/documents/upload"]["post"]
        upload_schema["requestBody"]["content"]["multipart/form-data"]["schema"] = {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string", "format": "binary"},
                }
            },
            "required": ["files"],
        }
    except KeyError:
        pass

    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            if not path.startswith("/auth") and path not in ["/", "/health"]:
                operation["security"] = [{"BearerAuth": []}]

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.on_event("startup")
async def startup():
    logger.info("Starting AI Document Assistant...")
    init_db()
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing.")
    logger.info("All systems ready.")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "app": "AI Document Assistant", "docs": "/docs"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model": config.GROQ_MODEL}