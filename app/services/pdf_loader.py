import fitz  # PyMuPDF
import os
from pathlib import Path
#from app.core.logger import get_logger

#logger = get_logger(__name__)


def extract_text_from_pdf(file_path: str) -> dict:
    
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    #logger.info(f"Opening PDF: {path.name}")

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ValueError(f"Could not open PDF '{path.name}': {e}")

    pages = []
    full_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # extract_text() returns a plain-text string for the page
        text = page.get_text("text").strip()

        if text:  # skip blank / image-only pages
            pages.append({
                "page_num": page_num + 1,   # 1-indexed for human readability
                "text": text,
            })
            full_text_parts.append(text)

    doc.close()

    if not pages:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The PDF may be scanned or image-based."
        )

    full_text = "\n\n".join(full_text_parts)

    #logger.info(
    #    f"Extracted {len(pages)} pages ({len(full_text)} chars) from '{path.name}'"
    #)

    return {
        "doc_id":      path.stem,          # e.g. "annual_report_2024"
        "file_name":   path.name,          # e.g. "annual_report_2024.pdf"
        "pages":       pages,
        "total_pages": len(pages),
        "full_text":   full_text,
    }


def get_pdf_metadata(file_path: str) -> dict:
    """
    Return lightweight metadata for a PDF without extracting all text.
    Useful for validation at upload time.

    Returns:
        { "title", "author", "page_count", "file_size_kb" }
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(str(path))
    meta = doc.metadata or {}
    page_count = len(doc)
    doc.close()

    file_size_kb = round(os.path.getsize(path) / 1024, 2)

    return {
        "title":        meta.get("title", ""),
        "author":       meta.get("author", ""),
        "page_count":   page_count,
        "file_size_kb": file_size_kb,
    }