import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from app.core import config
from app.core.logger import get_logger
from app.services.pdf_loader import extract_text_from_pdf
from app.services.chunker import chunk_pages
from app.services.embedding import embed_chunks
from app.services.vector_store import store_chunks, delete_document, list_documents, get_document_chunk_count

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    """
    Upload one or multiple PDF files and process each through the full RAG pipeline.

    Pipeline per file:
        Save → Extract text → Chunk → Embed → Store in ChromaDB

    Returns:
        List of results per file (success or error per file)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []

    for file in files:
        # ── Validate file type ─────────────────────────────────────────────
        if not file.filename.endswith(".pdf"):
            results.append({
                "filename": file.filename,
                "status":   "error",
                "detail":   "Only PDF files are accepted.",
            })
            continue

        # ── Validate file size ─────────────────────────────────────────────
        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > config.MAX_FILE_SIZE_MB:
            results.append({
                "filename": file.filename,
                "status":   "error",
                "detail":   f"File too large ({size_mb:.1f}MB). Max: {config.MAX_FILE_SIZE_MB}MB.",
            })
            continue

        # ── Generate unique doc_id and save file ───────────────────────────
        doc_id = str(uuid.uuid4())[:8] + "_" + os.path.splitext(file.filename)[0]
        doc_id = doc_id.replace(" ", "_")[:50]
        save_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}.pdf")

        with open(save_path, "wb") as f:
            f.write(contents)

        logger.info(f"Saved: {save_path}")

        # ── Run RAG pipeline ───────────────────────────────────────────────
        try:
            extracted    = extract_text_from_pdf(save_path)
            chunks       = chunk_pages(extracted["pages"], doc_id=doc_id, source_file=file.filename)
            chunks       = embed_chunks(chunks)
            stored_count = store_chunks(chunks)

            logger.info(f"Processed '{file.filename}' → doc_id='{doc_id}'")

            results.append({
                "status":       "success",
                "doc_id":       doc_id,
                "filename":     file.filename,
                "total_pages":  extracted["total_pages"],
                "total_chunks": stored_count,
                "size_mb":      round(size_mb, 2),
            })

        except Exception as e:
            os.remove(save_path)
            logger.error(f"Pipeline failed for '{file.filename}': {e}")
            results.append({
                "filename": file.filename,
                "status":   "error",
                "detail":   str(e),
            })

    success_count = sum(1 for r in results if r["status"] == "success")

    return {
        "message":       f"{success_count}/{len(files)} file(s) processed successfully.",
        "total_files":   len(files),
        "success_count": success_count,
        "results":       results,
    }


@router.get("/")
def list_all_documents():
    """List all documents currently stored in ChromaDB."""
    doc_ids = list_documents()

    docs = []
    for doc_id in doc_ids:
        docs.append({
            "doc_id":      doc_id,
            "chunk_count": get_document_chunk_count(doc_id),
        })

    return {
        "total_documents": len(docs),
        "documents":       docs,
    }


@router.delete("/{doc_id}")
def delete_doc(doc_id: str):
    """Delete a document and all its chunks from ChromaDB."""
    deleted = delete_document(doc_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    pdf_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        logger.info(f"Deleted PDF file: {pdf_path}")

    return {
        "message": f"Document '{doc_id}' deleted successfully.",
        "doc_id":  doc_id,
    }