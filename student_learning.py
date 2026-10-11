"""Grounded question generation, quiz grading, and weak-topic analysis."""

import json
import re
from collections import Counter
from urllib.request import Request, urlopen

from answer_generator import OLLAMA_MODEL, OLLAMA_URL, generate_grounded_response
from text_preprocessing import split_sentences, tokenize


QUESTION_TYPES = ("Short-answer", "Descriptive", "MCQ", "Mixed")
DIFFICULTIES = ("Easy", "Medium", "Hard")
MAX_QUESTIONS = 10
MAX_QUIZ_SOURCE_CHARS = 6000
QUIZ_REQUEST_TIMEOUT_SECONDS = 15

_ANSWER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "their", "this", "to",
    "was", "with",
}


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


def _normalize_question_type(question_type: str) -> str:
    return question_type.strip().casefold().replace("_", "-")


def _normalized_tokens(text: str) -> list[str]:
    return [token.casefold() for token in tokenize(text)]


def _has_source_evidence(source_text: str, evidence: str) -> bool:
    source_tokens = _normalized_tokens(source_text)
    evidence_tokens = _normalized_tokens(evidence)
    if not evidence_tokens:
        return False
    width = len(evidence_tokens)
    return any(
        source_tokens[index : index + width] == evidence_tokens
        for index in range(len(source_tokens) - width + 1)
    )


def _answer_is_supported(answer: str, evidence: str) -> bool:
    evidence_tokens = set(_normalized_tokens(evidence))
    answer_tokens = [
        token
        for token in _normalized_tokens(answer)
        if token not in _ANSWER_STOPWORDS or token in evidence_tokens
    ]
    if not answer_tokens:
        return False

    numeric_tokens = [token for token in answer_tokens if any(char.isdigit() for char in token)]
    if any(token not in evidence_tokens for token in numeric_tokens):
        return False

    overlap = sum(token in evidence_tokens for token in answer_tokens)
    minimum_overlap = max(1, (len(answer_tokens) + 1) // 2)
    return overlap >= minimum_overlap


def _verified_questions(
    generated: list, source_text: str, requested_type: str, limit: int
) -> list[dict]:
    verified_questions = []
    seen_questions = set()

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
        question_key = " ".join(_normalized_tokens(question))
        if (
            not question
            or question_key in seen_questions
            or not topic
            or not explanation
            or not correct_answer
            or not _has_source_evidence(source_text, evidence)
            or not _answer_is_supported(correct_answer, evidence)
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
                or not all(isinstance(option, str) and option.strip() for option in options)
                or len({option.strip().casefold() for option in options}) != 4
                or sum(option.strip().casefold() == correct_answer.casefold() for option in options) != 1
            ):
                continue
            item_result["options"] = [option.strip() for option in options]

        seen_questions.add(question_key)
        verified_questions.append(item_result)
        if len(verified_questions) == limit:
            break

    return verified_questions


def _generate_quiz_response(instructions: str, source_text: str) -> str:
    request_body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1400},
            "messages": [
                {
                    "role": "system",
                    "content": "Create concise quiz items using only the supplied passage. Return only JSON.",
                },
                {
                    "role": "user",
                    "content": f"{instructions}\n<passage>\n{source_text}\n</passage>",
                },
            ],
        }
    ).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=QUIZ_REQUEST_TIMEOUT_SECONDS) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result.get("message", {}).get("content", "").strip()


def _fallback_quiz(
    source_text: str, number_of_questions: int, question_type: str
) -> list[dict]:
    facts = []
    seen = set()
    for sentence in split_sentences(source_text):
        sentence = sentence.strip()
        sentence_key = " ".join(_normalized_tokens(sentence))
        if sentence_key in seen or len(tokenize(sentence)) < 2:
            continue
        seen.add(sentence_key)

        label_match = re.match(r"^\s*([^:]{1,60}):\s*(.+?)\s*[.!?]*$", sentence)
        relation_match = re.match(
            r"^\s*(.{1,80}?)\s+(is|are|was|were|has|have|uses|use|"
            r"completed|studied|studies|works|worked|developed|created|"
            r"includes|include|contains|contain)\s+(.+?)\s*[.!?]*$",
            sentence,
            re.IGNORECASE,
        )
        if label_match:
            label, answer = label_match.groups()
            question = f"What information is listed for {label.strip()}?"
            topic = label.strip()
            answer = answer.strip()
        elif relation_match:
            subject, verb, answer = relation_match.groups()
            subject = subject.strip()
            verb = verb.casefold()
            question_verbs = {
                "is": f"What is {subject}?",
                "are": f"What are {subject}?",
                "was": f"What was {subject}?",
                "were": f"What were {subject}?",
                "has": f"What does {subject} have?",
                "have": f"What do {subject} have?",
                "uses": f"What does {subject} use?",
                "use": f"What do {subject} use?",
                "completed": f"What did {subject} complete?",
                "studied": f"What did {subject} study?",
                "studies": f"What does {subject} study?",
                "works": f"Where does {subject} work?",
                "worked": f"Where did {subject} work?",
                "developed": f"What did {subject} develop?",
                "created": f"What did {subject} create?",
                "includes": f"What does {subject} include?",
                "include": f"What do {subject} include?",
                "contains": f"What does {subject} contain?",
                "contain": f"What do {subject} contain?",
            }
            question = question_verbs[verb]
            topic = subject
            answer = answer.strip()
        else:
            focus = " ".join(tokenize(sentence)[:4])
            question = f"What does the passage state about {focus}?"
            topic = focus
            answer = sentence

        facts.append(
            {
                "type": "Short-answer",
                "question": question,
                "topic": topic,
                "correct_answer": answer,
                "explanation": f"This fact is stated in the selected passage: {sentence}",
                "evidence": sentence,
            }
        )

    unique_facts = []
    seen_questions = set()
    for fact in facts:
        key = " ".join(_normalized_tokens(fact["question"]))
        if key not in seen_questions:
            unique_facts.append(fact)
            seen_questions.add(key)
        if len(unique_facts) == number_of_questions:
            break

    if question_type == "MCQ" or question_type == "Mixed":
        for index, fact in enumerate(unique_facts):
            if question_type == "Mixed" and index % 3 != 2:
                continue
            fact["type"] = "MCQ"
            distractors = [
                other["correct_answer"]
                for other in unique_facts
                if other is not fact
                and other["correct_answer"].casefold() != fact["correct_answer"].casefold()
            ]
            distractors.extend(
                [
                    "Not stated in the passage",
                    "Not specified by the source",
                    "Cannot be determined from the passage",
                ]
            )
            options = [fact["correct_answer"]]
            for distractor in distractors:
                if distractor.casefold() not in {option.casefold() for option in options}:
                    options.append(distractor)
                if len(options) == 4:
                    break
            fact["options"] = options

    if question_type == "Descriptive":
        for fact in unique_facts:
            fact["type"] = "Descriptive"
    elif question_type == "Mixed":
        for index, fact in enumerate(unique_facts):
            if fact["type"] != "MCQ" and index % 2:
                fact["type"] = "Descriptive"

    return unique_facts


def generate_quiz(
    source_text: str,
    number_of_questions: int = 5,
    difficulty: str = "Medium",
    question_type: str = "Mixed",
) -> list[dict]:
    """Generate source-grounded quiz items with a local fallback."""
    if not source_text.strip():
        raise ValueError("Select a document passage with extracted text first.")
    if not 1 <= number_of_questions <= MAX_QUESTIONS:
        raise ValueError(f"Choose between 1 and {MAX_QUESTIONS} questions.")
    if difficulty not in DIFFICULTIES or question_type not in QUESTION_TYPES:
        raise ValueError("Choose one of the available difficulty and question types.")

    quiz_source = source_text
    if len(quiz_source) > MAX_QUIZ_SOURCE_CHARS:
        quiz_source = quiz_source[:MAX_QUIZ_SOURCE_CHARS].rsplit(" ", 1)[0]
    requested_type = _normalize_question_type(question_type)
    type_instruction = (
        "Use a balanced mix of short-answer, descriptive, and MCQ questions."
        if requested_type == "mixed"
        else f"Every question type must be exactly {question_type}."
    )
    instructions = (
        f"Create up to {number_of_questions} {difficulty.lower()} quiz questions. "
        f"{type_instruction} Use only facts in the passage. Return JSON with a questions "
        "array; each item needs type, question, topic, correct_answer, explanation, and "
        "evidence. Evidence must be a short direct quote supporting the answer. Paraphrase "
        "is allowed. For MCQ, include four distinct options and the correct answer once."
    )

    verified_questions = []
    if len(tokenize(quiz_source)) >= 6:
        try:
            response = _generate_quiz_response(instructions, quiz_source)
            generated = _parse_json_response(response).get("questions", [])
            if isinstance(generated, list):
                verified_questions = _verified_questions(
                    generated, quiz_source, requested_type, number_of_questions
                )
        except Exception:
            pass

    if len(verified_questions) < number_of_questions:
        fallback_questions = _fallback_quiz(
            quiz_source,
            number_of_questions - len(verified_questions),
            question_type,
        )
        seen_questions = {
            " ".join(_normalized_tokens(item["question"]))
            for item in verified_questions
        }
        for item in fallback_questions:
            question_key = " ".join(_normalized_tokens(item["question"]))
            if question_key not in seen_questions:
                verified_questions.append(item)
                seen_questions.add(question_key)
            if len(verified_questions) == number_of_questions:
                break

    if not verified_questions:
        raise ValueError(
            "The selected source does not contain enough usable information to generate a quiz."
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