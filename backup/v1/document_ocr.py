"""OCR images and scanned PDF pages with the local Tesseract engine."""

import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pymupdf
from PIL import Image, ImageOps
import pytesseract

from document_ingestion import IMAGE_EXTENSIONS


OCR_LANGUAGE = "eng"
PDF_RENDER_DPI = 300
pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_CMD", pytesseract.pytesseract.tesseract_cmd
)


def _run_ocr(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")

    try:
        return pytesseract.image_to_string(image, lang=OCR_LANGUAGE)
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract OCR was not found. Install Tesseract and make sure it is on PATH."
        ) from error


def _extract_pdf_tables(page) -> list[str]:
    try:
        found_tables = page.find_tables()
    except Exception:
        return []

    tables = []
    for table_index, table in enumerate(found_tables.tables, start=1):
        rows = table.extract() or []
        rendered_rows = [
            " | ".join(str(cell or "").strip() for cell in row)
            for row in rows
        ]
        rendered_rows = [row for row in rendered_rows if row.strip(" |")]
        if rendered_rows:
            tables.append(f"Table {table_index}:\n" + "\n".join(rendered_rows))
    return tables


def _extract_pdf_image_text(pdf, page) -> list[str]:
    image_texts = []
    seen_xrefs = set()
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        pixmap = pymupdf.Pixmap(pdf, xref)
        if pixmap.n - pixmap.alpha > 3:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
        with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
            image_text = _run_ocr(image).strip()
        if image_text:
            image_texts.append(image_text)
    return image_texts


def extract_docx_image_text(file_bytes: bytes) -> list[str]:
    """OCR embedded DOCX images such as scanned figures and diagram labels."""
    image_texts = []
    with ZipFile(BytesIO(file_bytes)) as document_archive:
        image_paths = sorted(
            path
            for path in document_archive.namelist()
            if path.startswith("word/media/")
        )
        for image_path in image_paths:
            try:
                with Image.open(BytesIO(document_archive.read(image_path))) as image:
                    image_text = _run_ocr(image).strip()
            except (OSError, ValueError):
                continue
            if image_text:
                image_texts.append(image_text)
    return image_texts


def extract_ocr_pages(file_name: str, file_bytes: bytes) -> list[dict]:
    """Return page-numbered text from a PDF or text from a supported image.

    PDFs retain their text layer, extract detected tables, and OCR embedded image
    blocks. Pages without a text layer receive full-page OCR. Images are one page.
    """
    extension = Path(file_name).suffix.lower().lstrip(".")

    if extension == "pdf":
        pages = []
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text")
                if not text.strip():
                    pixmap = page.get_pixmap(dpi=PDF_RENDER_DPI, alpha=False)
                    with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
                        text = _run_ocr(image)
                else:
                    additional_parts = _extract_pdf_tables(page)
                    additional_parts.extend(_extract_pdf_image_text(pdf, page))
                    text = "\n\n".join([text.strip(), *additional_parts])
                pages.append({"page_number": page_index + 1, "text": text})
        return pages

    if extension in IMAGE_EXTENSIONS:
        with Image.open(BytesIO(file_bytes)) as image:
            return [{"page_number": None, "text": _run_ocr(image)}]

    raise ValueError("OCR supports PDF files and common image formats.")