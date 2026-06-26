import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
CHROMA_DB_PATH   = "chroma_db"
UPLOAD_DIR       = "uploads"
LOG_DIR          = "logs"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
TOP_K_RESULTS    = 5
MAX_FILE_SIZE_MB = 20
LOG_LEVEL        = "INFO"

# Auto-create folders on import
for _dir in [UPLOAD_DIR, CHROMA_DB_PATH, LOG_DIR]:
    Path(_dir).mkdir(parents=True, exist_ok=True)