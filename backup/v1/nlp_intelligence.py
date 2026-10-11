"""Extractive summaries, TF-IDF keywords, categories, and document analytics."""

from collections import Counter
import math
import re

from text_preprocessing import normalize_text, split_sentences, tokenize


_STOP_WORDS = frozenset(
    "a about above after again against all am an and any are as at be because been "
    "before being below between both but by can could did do does doing down during "
    "each few for from further had has have having he her here hers herself him himself "
    "his how i if in into is it its itself just me more most my myself no nor not of off "
    "on once only or other our ours ourselves out over own same she should so some such "
    "than that the their theirs them themselves then there these they this those through "
    "to too under until up very was we were what when where which while who whom why with "
    "would you your yours yourself yourselves"
    .split()
)

_CATEGORY_TERMS = {
    "Academic": (
        "academic", "research", "journal", "scholarly", "study", "theory", "analysis",
        "publication", "literature review",
    ),
    "Examination": (
        "exam", "examination", "test", "question paper", "marks", "grade", "hall ticket",
        "exam timetable", "assessment",
    ),
    "Placement": (
        "placement", "recruitment", "career", "interview", "resume", "company", "employer",
        "internship", "job offer",
    ),
    "Circular": (
        "circular", "circular notice", "hereby", "all students", "all faculty", "administrative",
    ),
    "Syllabus": (
        "syllabus", "course outline", "curriculum", "learning outcome", "course objective",
        "credit", "unit ", "module",
    ),
    "Notes": (
        "notes", "lecture", "chapter", "definition", "concept", "example", "key point",
    ),
    "Assignment": (
        "assignment", "homework", "submission", "submit", "due date", "project work",
        "problem set", "coursework",
    ),
    "Department Notice": (
        "department notice", "department", "faculty meeting", "department meeting", "venue",
        "event", "students are informed", "notice is hereby",
    ),
}
_CATEGORY_ORDER = tuple(_CATEGORY_TERMS)
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def _content_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in tokenize(text)
        if token.casefold() not in _STOP_WORDS and (len(token) > 1 or token.isdigit())
    ]


def _document_statistics(sentences: list[str]) -> tuple[Counter, dict[str, float]]:
    sentence_tokens = [_content_tokens(sentence) for sentence in sentences]
    frequencies = Counter(token for tokens in sentence_tokens for token in tokens)
    document_frequencies = Counter(
        token for tokens in sentence_tokens for token in set(tokens)
    )
    sentence_count = max(1, len(sentence_tokens))
    tfidf = {
        token: (1 + math.log(count))
        * (math.log((1 + sentence_count) / (1 + document_frequencies[token])) + 1)
        for token, count in frequencies.items()
    }
    return frequencies, tfidf


def _sentence_score(sentence: str, tfidf: dict[str, float]) -> float:
    terms = _content_tokens(sentence)
    if not terms:
        return 0.0
    return sum(tfidf.get(term, 0.0) for term in terms) / math.sqrt(len(terms))


def _ranked_sentence_summary(
    sentences: list[str], tfidf: dict[str, float], limit: int
) -> list[str]:
    if not sentences:
        return []
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (_sentence_score(sentences[index], tfidf), -index),
        reverse=True,
    )
    selected = set(ranked[:limit])
    return [sentence for index, sentence in enumerate(sentences) if index in selected]


def _extract_keyphrases(text: str, tfidf: dict[str, float]) -> list[str]:
    words = [
        token.casefold()
        for token in _WORD_PATTERN.findall(text)
        if token.casefold() not in _STOP_WORDS and (len(token) > 1 or token.isdigit())
    ]
    phrase_scores = Counter()
    for phrase_length in (2, 3):
        for index in range(len(words) - phrase_length + 1):
            phrase_words = words[index : index + phrase_length]
            phrase = " ".join(phrase_words)
            phrase_scores[phrase] += sum(tfidf.get(word, 0.0) for word in phrase_words)

    ranked_phrases = sorted(
        phrase_scores,
        key=lambda phrase: (phrase_scores[phrase], len(phrase.split()), phrase),
        reverse=True,
    )
    selected = []
    for phrase in ranked_phrases:
        if any(phrase in existing or existing in phrase for existing in selected):
            continue
        selected.append(phrase)
        if len(selected) == 5:
            break
    return selected


def _classify(text: str, tfidf: dict[str, float]) -> tuple[str, int]:
    normalized = f" {normalize_text(text).casefold()} "
    scores = {}
    matched_counts = {}
    for category, terms in _CATEGORY_TERMS.items():
        matched = [
            term
            for term in terms
            if f" {term.strip().casefold()} " in normalized
        ]
        scores[category] = sum(
            1.0 + sum(tfidf.get(token, 0.0) for token in _content_tokens(term))
            for term in matched
        )
        matched_counts[category] = len(matched)

    ordered = sorted(
        _CATEGORY_ORDER,
        key=lambda category: (scores[category], matched_counts[category]),
        reverse=True,
    )
    best_category = ordered[0]
    best_score = scores[best_category]
    if best_score == 0:
        return "Unclassified", 0

    second_score = scores[ordered[1]]
    evidence = min(1.0, matched_counts[best_category] / 3)
    separation = (best_score - second_score) / best_score
    confidence = round(100 * (0.65 * evidence + 0.35 * separation))
    if matched_counts[best_category] == 1:
        confidence = min(confidence, 55)
    elif matched_counts[best_category] == 2:
        confidence = min(confidence, 75)
    return best_category, confidence


def analyze_document(text: str, page_count: int | None = None) -> dict:
    """Compute extractive summaries, keywords, category, and descriptive counts."""
    cleaned_text = normalize_text(text)
    sentences = split_sentences(cleaned_text)
    words = tokenize(cleaned_text)
    frequencies, tfidf = _document_statistics(sentences)

    short_summary = _ranked_sentence_summary(sentences, tfidf, limit=3)
    detailed_limit = min(8, max(3, math.ceil(len(sentences) * 0.4)))
    detailed_summary = _ranked_sentence_summary(sentences, tfidf, detailed_limit)
    bullet_summary = _ranked_sentence_summary(sentences, tfidf, limit=5)

    keywords = [
        term
        for term, _ in sorted(
            tfidf.items(), key=lambda item: (item[1], frequencies[item[0]], item[0]), reverse=True
        )[:10]
    ]
    keyphrases = _extract_keyphrases(cleaned_text, tfidf)
    category, classification_confidence = _classify(cleaned_text, tfidf)

    return {
        "page_count": page_count,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "short_summary": " ".join(short_summary),
        "detailed_summary": " ".join(detailed_summary),
        "bullet_summary": bullet_summary,
        "category": category,
        "classification_confidence": classification_confidence,
        "major_topics": keyphrases,
        "important_keywords": keywords,
        "frequent_terms": [
            {"term": term, "count": count}
            for term, count in frequencies.most_common(10)
        ],
    }