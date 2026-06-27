from typing import Optional
from app.core.logger import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    source_file: Optional[str] = None,
) -> list[dict]:
   
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
        )

    text = text.strip()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take whatever remains
            chunk_text_slice = text[start:]
        else:
            # Try to cut at the nearest sentence boundary (. ! ?)
            # to avoid splitting mid-sentence
            boundary = _find_sentence_boundary(text, end)
            chunk_text_slice = text[start:boundary]
            end = boundary

        chunk_text_slice = chunk_text_slice.strip()

        if chunk_text_slice:
            chunks.append({
                "chunk_id":    f"{doc_id}_{chunk_index}",
                "doc_id":      doc_id,
                "text":        chunk_text_slice,
                "char_start":  start,
                "char_end":    start + len(chunk_text_slice),
                "chunk_index": chunk_index,
                "source_file": source_file or "",
            })
            chunk_index += 1

        # Move start forward by (chunk_size - overlap) to create the sliding window
        start = end - chunk_overlap

        # Safety: prevent infinite loop if boundary didn't advance
        if start <= 0 and chunk_index > 0:
            break

    logger.info(
        f"Chunked doc '{doc_id}' → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks


def chunk_pages(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    source_file: Optional[str] = None,
) -> list[dict]:
    
    all_chunks = []
    global_index = 0

    for page in pages:
        page_num = page["page_num"]
        page_text = page["text"].strip()

        if not page_text:
            continue

        page_chunks = chunk_text(
            text=page_text,
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_file=source_file,
        )

        for chunk in page_chunks:
            chunk["chunk_id"] = f"{doc_id}_{global_index}"
            chunk["chunk_index"] = global_index
            chunk["page_num"] = page_num          # ← page-level source reference
            all_chunks.append(chunk)
            global_index += 1

    logger.info(
        f"Page-chunked doc '{doc_id}' → {len(all_chunks)} total chunks "
        f"across {len(pages)} pages"
    )

    return all_chunks


def _find_sentence_boundary(text: str, position: int, search_window: int = 100) -> int:
    
    search_start = max(0, position - search_window)
    snippet = text[search_start:position]

    # Walk backwards looking for sentence-ending chars
    for i in range(len(snippet) - 1, -1, -1):
        if snippet[i] in ".!?":
            return search_start + i + 1     # position right after the punctuation

    return position     # fallback: no boundary found, cut at hard limit