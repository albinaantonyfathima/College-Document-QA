"""Persist processed chunks as JSON for the next NLP/RAG module."""

import json
from pathlib import Path
import re


PROCESSED_DIRECTORY = Path(__file__).resolve().parent / "data" / "processed"
_STORED_NAME_PATTERN = re.compile(r"([a-f0-9]{32})\.[a-z0-9]+\Z")


def save_processed_chunks(stored_name: str, chunks: list[dict]) -> str:
    """Save a document's chunks as UTF-8 JSON and return its project-relative path."""
    match = _STORED_NAME_PATTERN.fullmatch(stored_name)
    if not match:
        raise ValueError("Invalid generated storage name.")

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIRECTORY / f"{match.group(1)}.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {"schema_version": 1, "chunks": chunks},
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    return (Path("data") / "processed" / output_path.name).as_posix()