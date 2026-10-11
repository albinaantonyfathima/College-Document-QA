"""Coordinate extraction, OCR, preprocessing, and chunk creation."""

from pathlib import Path

from document_ingestion import IMAGE_EXTENSIONS, extract_text
from document_ocr import extract_docx_image_text, extract_ocr_pages
from text_chunking import chunk_pages
from text_preprocessing import normalize_text


def process_document(file_name: str, file_bytes: bytes) -> tuple[list[dict], list[dict]]:
    """Extract and clean text, then return source pages and retrieval-ready chunks."""
    extension = Path(file_name).suffix.lower().lstrip(".")
    if extension == "pdf" or extension in IMAGE_EXTENSIONS:
        pages = extract_ocr_pages(file_name, file_bytes)
    elif extension == "docx":
        text = extract_text(file_name, file_bytes)
        image_text = extract_docx_image_text(file_bytes)
        if image_text:
            text = "\n\n".join([text, "Embedded image text:", *image_text])
        pages = [{"page_number": None, "text": text}]
    else:
        pages = [{"page_number": None, "text": extract_text(file_name, file_bytes)}]

    cleaned_pages = [
        {"page_number": page["page_number"], "text": normalize_text(page["text"])}
        for page in pages
    ]
    chunks = chunk_pages(cleaned_pages, document_name=file_name)
    return cleaned_pages, chunks