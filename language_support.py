"""Lightweight English/Tamil question language detection and answer selection."""

import re


ANSWER_LANGUAGE_OPTIONS = ("Auto-detect", "English", "Tamil")
_TAMIL_SCRIPT_PATTERN = re.compile(r"[\u0B80-\u0BFF]")


def detect_question_language(question: str) -> str:
    """Detect Tamil by script; treat other text as English for this two-language UI."""
    if _TAMIL_SCRIPT_PATTERN.search(question):
        return "Tamil"
    return "English"


def select_answer_language(question: str, preference: str) -> tuple[str, str]:
    """Return (detected question language, answer language)."""
    detected_language = detect_question_language(question)
    if preference == "Auto-detect":
        return detected_language, detected_language
    if preference not in {"English", "Tamil"}:
        raise ValueError("Choose Auto-detect, English, or Tamil.")
    return detected_language, preference