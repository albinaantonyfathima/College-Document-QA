"""NLP text normalization, sentence splitting, and word tokenization."""

import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"'’”)]*\s+|\n{2,}")


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while retaining paragraph boundaries."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "").replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        character
        for character in text
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation and blank paragraph boundaries."""
    normalized = normalize_text(text)
    return [
        part.strip()
        for part in _SENTENCE_BOUNDARY.split(normalized)
        if part.strip()
    ]


def tokenize(text: str) -> list[str]:
    """Return word and number tokens, keeping internal apostrophes together."""
    return _TOKEN_PATTERN.findall(text)