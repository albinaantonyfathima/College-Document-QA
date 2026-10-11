"""Create document embeddings and retrieve relevant chunks with Chroma."""

import json
from functools import lru_cache
from pathlib import Path
import re

import chromadb
from fastembed import TextEmbedding

from processed_store import PROCESSED_DIRECTORY


VECTOR_DIRECTORY = Path(__file__).resolve().parent / "data" / "vector_db"
COLLECTION_NAME = "college_document_chunks_multilingual_v1"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5
MAX_COSINE_DISTANCE = 0.55
MIN_RETRIEVAL_SIMILARITY = 1.0 - MAX_COSINE_DISTANCE
HIGH_CONFIDENCE_SIMILARITY = 0.75
MAX_CONFIDENCE_SIMILARITY = 0.80
_DOCUMENT_ID_PATTERN = re.compile(r"[a-f0-9]{32}\Z")


@lru_cache(maxsize=1)
def _get_embedding_model():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    VECTOR_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DIRECTORY))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def _embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _get_embedding_model().embed(texts)
    return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]


def semantic_similarity(first_text: str, second_text: str) -> float:
    """Return cosine similarity using the same embedding model as document retrieval."""
    if not first_text.strip() or not second_text.strip():
        return 0.0

    first_vector, second_vector = _embed_texts([first_text, second_text])
    dot_product = sum(first * second for first, second in zip(first_vector, second_vector))
    first_norm = sum(value * value for value in first_vector) ** 0.5
    second_norm = sum(value * value for value in second_vector) ** 0.5
    if not first_norm or not second_norm:
        return 0.0
    return max(-1.0, min(1.0, dot_product / (first_norm * second_norm)))


def index_chunks(document_id: str, chunks: list[dict]) -> int:
    """Embed and upsert one document's chunks; return the number indexed."""
    if not _DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise ValueError("Document ID must be the generated 32-character storage ID.")

    valid_chunks = [chunk for chunk in chunks if chunk.get("text", "").strip()]
    if not valid_chunks:
        return 0

    collection = _get_collection()
    collection.upsert(
        ids=[f"{document_id}:{chunk['chunk_number']}" for chunk in valid_chunks],
        embeddings=_embed_texts([chunk["text"] for chunk in valid_chunks]),
        documents=[chunk["text"] for chunk in valid_chunks],
        metadatas=[
            {
                "document_id": document_id,
                "document_name": str(chunk["document_name"]),
                "page_number": (
                    int(chunk["page_number"])
                    if chunk["page_number"] is not None
                    else -1
                ),
                "chunk_number": int(chunk["chunk_number"]),
                "token_count": int(chunk.get("token_count", 0)),
            }
            for chunk in valid_chunks
        ],
    )
    return len(valid_chunks)


def retrieve_relevant_chunks(
    question: str,
    top_k: int = TOP_K,
    max_distance: float = MAX_COSINE_DISTANCE,
) -> list[dict]:
    """Embed a question and return its nearest chunks under the cosine threshold."""
    if not question.strip() or top_k < 1:
        return []

    collection = _get_collection()
    collection_size = collection.count()
    if collection_size == 0:
        return []

    question_embedding = _embed_texts([question])[0]
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection_size),
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        distance = float(distance)
        if distance > max_distance:
            continue
        page_number = int(metadata["page_number"])
        matches.append(
            {
                "document_name": metadata["document_name"],
                "page_number": page_number if page_number >= 1 else None,
                "chunk_number": int(metadata["chunk_number"]),
                "text": text,
                "distance": distance,
                "similarity": 1.0 - distance,
            }
        )
    return matches


def assess_retrieval_confidence(chunks: list[dict], cross_lingual: bool = False) -> dict:
    """Estimate retrieval quality from cosine similarity; this is not model certainty."""
    if not chunks:
        return {"score": 0, "label": "No match", "best_similarity": 0.0}

    best_similarity = max(float(chunk["similarity"]) for chunk in chunks)
    score_floor = MIN_RETRIEVAL_SIMILARITY
    score_ceiling = 0.65 if cross_lingual else MAX_CONFIDENCE_SIMILARITY
    score = round(
        100
        * max(
            0.0,
            min(
                1.0,
                (best_similarity - score_floor) / (score_ceiling - score_floor),
            ),
        )
    )

    if best_similarity >= HIGH_CONFIDENCE_SIMILARITY and score >= 75:
        label = "High"
    elif (cross_lingual and best_similarity >= 0.50) or (not cross_lingual and score >= 40):
        label = "Moderate"
    else:
        label = "Low"

    return {
        "score": score,
        "label": label,
        "best_similarity": best_similarity,
    }


def reindex_saved_chunks() -> int:
    """Index JSON chunks saved by the previous processing batch."""
    if not PROCESSED_DIRECTORY.exists():
        return 0

    indexed_count = 0
    for path in sorted(PROCESSED_DIRECTORY.glob("*.json")):
        if not _DOCUMENT_ID_PATTERN.fullmatch(path.stem):
            continue
        with path.open("r", encoding="utf-8") as processed_file:
            saved_data = json.load(processed_file)
        indexed_count += index_chunks(path.stem, saved_data.get("chunks", []))
    return indexed_count