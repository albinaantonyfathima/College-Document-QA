"""Compare two processed documents using topics, semantic embeddings, and text diffs."""

from collections import Counter
from difflib import SequenceMatcher
import re

from nlp_intelligence import analyze_document
from rag_retriever import semantic_similarity
from text_preprocessing import normalize_text, split_sentences


_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(
        r"\b(?:rule|regulation|section|clause|article|policy)\s*(?:no\.?\s*)?"
        r"\d+(?:\.\d+)*(?:[a-z])?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:\.\d+)?\s?(?:%|(?:percent|credits?|marks?|days?|weeks?|hours?)\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:19|20)\d{2}\b"),
)


def _topics(text: str, provided_topics: list[str] | None) -> list[str]:
    if provided_topics is not None:
        return provided_topics
    return analyze_document(text)["major_topics"]


def _extract_values(text: str) -> Counter:
    values = Counter()
    for pattern in _VALUE_PATTERNS:
        values.update(match.group(0) for match in pattern.finditer(text))
    return values


def _sentence_changes(first_text: str, second_text: str) -> list[dict]:
    first_sentences = split_sentences(normalize_text(first_text))
    second_sentences = split_sentences(normalize_text(second_text))
    matcher = SequenceMatcher(
        None,
        [sentence.casefold() for sentence in first_sentences],
        [sentence.casefold() for sentence in second_sentences],
        autojunk=False,
    )
    changes = []
    for operation, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if operation in {"replace", "delete"}:
            changes.extend(
                {"change": "Removed from first document", "text": sentence}
                for sentence in first_sentences[first_start:first_end]
            )
        if operation in {"replace", "insert"}:
            changes.extend(
                {"change": "Added in second document", "text": sentence}
                for sentence in second_sentences[second_start:second_end]
            )
    return changes[:20]


def compare_documents(
    first_name: str,
    first_text: str,
    second_name: str,
    second_text: str,
    first_topics: list[str] | None = None,
    second_topics: list[str] | None = None,
) -> dict:
    """Compare topics, semantic similarity, sentence changes, and numeric markers."""
    first_normalized = normalize_text(first_text)
    second_normalized = normalize_text(second_text)
    if not first_normalized or not second_normalized:
        raise ValueError("Both documents need extracted text before comparison.")

    first_topics = _topics(first_normalized, first_topics)
    second_topics = _topics(second_normalized, second_topics)
    first_topic_map = {topic.casefold(): topic for topic in first_topics}
    second_topic_map = {topic.casefold(): topic for topic in second_topics}
    common_topics = [
        topic for topic in first_topics if topic.casefold() in second_topic_map
    ]
    first_unique_topics = [
        topic for key, topic in first_topic_map.items() if key not in second_topic_map
    ]
    second_unique_topics = [
        topic for key, topic in second_topic_map.items() if key not in first_topic_map
    ]

    first_values = _extract_values(first_normalized)
    second_values = _extract_values(second_normalized)
    first_value_map = {value.casefold(): value for value in first_values}
    second_value_map = {value.casefold(): value for value in second_values}

    return {
        "first_document": first_name,
        "second_document": second_name,
        "similarity_score": round(
            100 * semantic_similarity(first_normalized, second_normalized)
        ),
        "common_topics": common_topics,
        "first_unique_topics": first_unique_topics,
        "second_unique_topics": second_unique_topics,
        "sentence_changes": _sentence_changes(first_normalized, second_normalized),
        "first_only_values": [
            first_value_map[key]
            for key in first_value_map.keys() - second_value_map
        ],
        "second_only_values": [
            second_value_map[key]
            for key in second_value_map.keys() - first_value_map
        ],
    }