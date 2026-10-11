"""Grounded question generation, quiz grading, and weak-topic analysis."""

import json
import re
from collections import Counter

from answer_generator import generate_grounded_response
from text_preprocessing import tokenize


QUESTION_TYPES = ("Short-answer", "Descriptive", "MCQ", "Mixed")
DIFFICULTIES = ("Easy", "Medium", "Hard")
MAX_QUESTIONS = 10


def explain_simply(source_text: str) -> str:
    """Explain a selected passage in clear language without adding outside facts."""
    if not source_text.strip():
        raise ValueError("Select document content with extracted text first.")

    explanation = generate_grounded_response(
        "Explain this passage for a student using short sentences and familiar words. "
        "Keep the original meaning and all important names, numbers, and relationships. "
        "Define technical terms only using information in the passage. Do not add facts, "
        "examples, analogies, or advice that are not supported by the passage. If a "
        "point is unclear in the passage, say that it is unclear.",
        source_text,
    )
    if not explanation:
        raise ValueError("The explanation model returned no text.")
    return explanation


def _parse_json_response(response: str) -> dict:
    response = response.strip()
    response = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", response)
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The question generator did not return valid JSON.")
    try:
        return json.loads(response[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError("The question generator returned malformed JSON.") from error


def _is_source_phrase(source_text: str, phrase: str) -> bool:
    return bool(phrase and phrase.casefold() in source_text.casefold())


def _normalize_question_type(question_type: str) -> str:
    return question_type.strip().casefold().replace("_", "-")


def generate_quiz(
    source_text: str,
    number_of_questions: int = 5,
    difficulty: str = "Medium",
    question_type: str = "Mixed",
) -> list[dict]:
    """Generate quiz items and retain only those verifiable against source text."""
    if not source_text.strip():
        raise ValueError("Select a document passage with extracted text first.")
    if not 1 <= number_of_questions <= MAX_QUESTIONS:
        raise ValueError(f"Choose between 1 and {MAX_QUESTIONS} questions.")
    if difficulty not in DIFFICULTIES or question_type not in QUESTION_TYPES:
        raise ValueError("Choose one of the available difficulty and question types.")

    requested_type = _normalize_question_type(question_type)
    type_instruction = (
        "Use a balanced mix of short-answer, descriptive, and MCQ questions."
        if requested_type == "mixed"
        else f"Every question type must be exactly {requested_type}."
    )
    instructions = f"""
Create exactly {number_of_questions} {difficulty.lower()}-difficulty quiz questions.
{type_instruction}
Return only a JSON object with a `questions` array. Every item must have:
`type`, `question`, `topic`, `correct_answer`, `explanation`, and `evidence`.
For MCQs also include exactly four `options`, with exactly one correct option.
The `topic`, `correct_answer`, `explanation`, `evidence`, and every MCQ option must
be exact phrases copied from the passage, including spelling and capitalization.
The evidence must directly support the correct answer. Write the question using only
information stated in the passage; do not ask for outside knowledge or unsupported
inference. Use one of these exact type values: Short-answer, Descriptive, MCQ.
For Hard, test relationships or distinctions explicitly described by the passage;
never require facts that are not written in it.
""".strip()
    response = generate_grounded_response(instructions, source_text)
    generated = _parse_json_response(response).get("questions", [])
    if not isinstance(generated, list):
        raise ValueError("The question generator returned an invalid question list.")

    verified_questions = []
    for item in generated:
        if not isinstance(item, dict):
            continue
        question_kind = str(item.get("type", "")).strip()
        if question_kind not in QUESTION_TYPES[:-1]:
            continue
        if requested_type != "mixed" and _normalize_question_type(question_kind) != requested_type:
            continue

        question = str(item.get("question", "")).strip()
        topic = str(item.get("topic", "")).strip()
        correct_answer = str(item.get("correct_answer", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not question or not all(
            _is_source_phrase(source_text, phrase)
            for phrase in (topic, correct_answer, explanation, evidence)
        ):
            continue

        item_result = {
            "type": question_kind,
            "question": question,
            "topic": topic,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "evidence": evidence,
        }
        if question_kind == "MCQ":
            options = item.get("options")
            if (
                not isinstance(options, list)
                or len(options) != 4
                or not all(
                    isinstance(option, str)
                    and _is_source_phrase(source_text, option.strip())
                    for option in options
                )
                or sum(option.strip().casefold() == correct_answer.casefold() for option in options) != 1
            ):
                continue
            item_result["options"] = [option.strip() for option in options]
        verified_questions.append(item_result)
        if len(verified_questions) == number_of_questions:
            break

    if not verified_questions:
        raise ValueError(
            "No generated questions passed source verification. Try another passage or retry."
        )
    return verified_questions


def _answer_matches(response: str, question: dict) -> bool:
    response = response.strip()
    if question["type"] == "MCQ":
        return response.casefold() == question["correct_answer"].casefold()
    if not response:
        return False

    response_tokens = {token.casefold() for token in tokenize(response)}
    expected_tokens = {token.casefold() for token in tokenize(question["correct_answer"])}
    return bool(expected_tokens) and expected_tokens.issubset(response_tokens)


def grade_quiz(questions: list[dict], responses: dict[int, str]) -> dict:
    """Score submitted answers and identify topics associated with incorrect ones."""
    results = []
    weak_topics = Counter()
    for index, question in enumerate(questions):
        user_answer = str(responses.get(index, "")).strip()
        correct = _answer_matches(user_answer, question)
        if not correct:
            weak_topics[question["topic"]] += 1
        results.append(
            {
                "question": question["question"],
                "user_answer": user_answer or "No answer submitted",
                "correct_answer": question["correct_answer"],
                "explanation": question["explanation"],
                "evidence": question["evidence"],
                "topic": question["topic"],
                "correct": correct,
            }
        )

    correct_count = sum(result["correct"] for result in results)
    total = len(results)
    return {
        "correct": correct_count,
        "total": total,
        "score_percent": round(100 * correct_count / total) if total else 0,
        "results": results,
        "weak_topics": [topic for topic, _ in weak_topics.most_common()],
    }