"""Safely store uploaded source documents on the local filesystem."""

import os
from pathlib import Path
from uuid import uuid4

from document_ingestion import SUPPORTED_EXTENSIONS


UPLOAD_DIRECTORY = Path(__file__).resolve().parent / "data" / "uploads"
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


def store_document(file_name: str, content: bytes) -> str:
    """Store a supported document under a generated name and return that name."""
    extension = Path(file_name).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(f"Unsupported file type. Supported formats: {supported}.")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        limit_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise ValueError(f"File exceeds the {limit_mb} MiB upload limit.")

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}.{extension}"
    destination = UPLOAD_DIRECTORY / stored_name
    file_descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )

    try:
        with os.fdopen(file_descriptor, "wb") as stored_file:
            stored_file.write(content)
    except OSError:
        destination.unlink(missing_ok=True)
        raise

    return stored_name