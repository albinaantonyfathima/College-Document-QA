"""College Document Q&A System: document upload and text extraction module."""

import hashlib

import streamlit as st

from document_ingestion import SUPPORTED_EXTENSIONS
from document_store import MAX_UPLOAD_SIZE_BYTES, store_document
from document_processing import process_document
from processed_store import save_processed_chunks
from answer_generator import (
    NO_ANSWER,
    generate_answer,
    resolve_followup_question,
    translate_question_for_retrieval,
)
from rag_retriever import (
    assess_retrieval_confidence,
    index_chunks,
    reindex_saved_chunks,
    retrieve_relevant_chunks,
)
from nlp_intelligence import analyze_document
from student_learning import (
    DIFFICULTIES,
    MAX_QUESTIONS,
    QUESTION_TYPES,
    explain_simply,
    generate_quiz,
    grade_quiz,
)
from document_comparison import compare_documents
from language_support import ANSWER_LANGUAGE_OPTIONS, select_answer_language
from streamlit_mic_recorder import speech_to_text
from voice_output import speak_answer


def _render_sources(sources: list[dict], note: str = "") -> None:
    with st.expander("Inspect retrieved sources"):
        if note:
            st.caption(note)
        if not sources:
            st.info("No source chunks were retrieved for this answer.")
            return

        for index, source in enumerate(sources, start=1):
            page = (
                f", page {source['page_number']}"
                if source["page_number"] is not None
                else ""
            )
            st.markdown(f"**[{index}] {source['document_name']}{page}**")
            st.caption(
                f"Chunk {source['chunk_number']} · "
                f"retrieval similarity {source['similarity']:.2f}"
            )
            st.text(source["text"])


def _render_confidence(confidence: dict) -> None:
    st.caption(
        f"Retrieval confidence: {confidence['score']}% · {confidence['label']} "
        "(heuristic, not model certainty)"
    )
    st.progress(confidence["score"] / 100)


st.set_page_config(
    page_title="College Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("College Document Q&A System")
st.caption("Intelligent Academic Assistant · Document ingestion")
st.write(
    "Upload course documents to store them locally, extract their text, and prepare "
    "page-aware chunks. Images and scanned PDF pages use local Tesseract OCR."
)

uploaded_files = st.file_uploader(
    "Choose documents",
    type=list(SUPPORTED_EXTENSIONS),
    accept_multiple_files=True,
    help=(
        f"Each file can be up to {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MiB. "
        "Handwriting recognition is best-effort and depends on the source quality."
    ),
)

documents = st.session_state.setdefault("documents", {})
RAG_INDEX_VERSION = 2
if st.session_state.get("rag_index_version") != RAG_INDEX_VERSION:
    try:
        reindex_saved_chunks()
    except Exception as error:
        st.warning(f"Could not index previously processed documents: {error}")
    else:
        st.session_state["rag_index_version"] = RAG_INDEX_VERSION

for uploaded_file in uploaded_files or []:
    content = uploaded_file.getvalue()
    fingerprint = hashlib.sha256(
        uploaded_file.name.encode("utf-8") + b"\0" + content
    ).hexdigest()
    if fingerprint in documents:
        continue

    record = {
        "filename": uploaded_file.name,
        "type": uploaded_file.name.rsplit(".", 1)[-1].upper()
        if "." in uploaded_file.name
        else "Unknown",
        "size": len(content),
        "status": "Upload failed",
        "stored_name": None,
        "text": "",
        "chunks": [],
        "processed_path": None,
        "page_count": None,
        "analytics": None,
        "error": "",
    }

    try:
        record["stored_name"] = store_document(uploaded_file.name, content)
    except (OSError, ValueError) as error:
        record["error"] = str(error)
    else:
        try:
            pages, record["chunks"] = process_document(uploaded_file.name, content)
            record["text"] = "\n\n".join(page["text"] for page in pages if page["text"])
            page_numbers = [
                page["page_number"]
                for page in pages
                if page["page_number"] is not None
            ]
            record["page_count"] = max(page_numbers) if page_numbers else None
            record["processed_path"] = save_processed_chunks(
                record["stored_name"], record["chunks"]
            )
        except Exception as error:
            record["status"] = "Stored; processing failed"
            record["error"] = str(error)
        else:
            if record["chunks"]:
                try:
                    indexed_count = index_chunks(
                        record["stored_name"].rsplit(".", 1)[0], record["chunks"]
                    )
                except Exception as error:
                    record["status"] = "Processed; vector indexing failed"
                    record["error"] = str(error)
                else:
                    record["status"] = f"Stored; indexed {indexed_count} chunks"
            else:
                record["status"] = "Stored; no text found"

    documents[fingerprint] = record

for record in documents.values():
    if record.get("text") and record.get("analytics") is None:
        record["analytics"] = analyze_document(
            record["text"], page_count=record.get("page_count")
        )

if not documents:
    st.info("Upload one or more PDF, DOCX, or TXT files to get started.")
else:
    st.subheader("Uploaded documents")
    st.dataframe(
        [
            {
                "Filename": record["filename"],
                "Type": record["type"],
                "Size": f"{record['size']:,} bytes",
                "Upload status": record["status"],
            }
            for record in documents.values()
        ],
        hide_index=True,
        use_container_width=True,
    )

    analyzed_documents = [
        record
        for record in documents.values()
        if record.get("analytics") is not None
    ]
    if analyzed_documents:
        st.subheader("NLP Analytics Dashboard")
        total_words = sum(record["analytics"]["word_count"] for record in analyzed_documents)
        total_sentences = sum(
            record["analytics"]["sentence_count"] for record in analyzed_documents
        )
        metric_columns = st.columns(3)
        metric_columns[0].metric("Analyzed documents", len(analyzed_documents))
        metric_columns[1].metric("Total words", f"{total_words:,}")
        metric_columns[2].metric("Total sentences", f"{total_sentences:,}")
        st.dataframe(
            [
                {
                    "Document": record["filename"],
                    "Category": record["analytics"]["category"],
                    "Classification confidence": (
                        f"{record['analytics']['classification_confidence']}%"
                    ),
                    "Pages": record["analytics"]["page_count"] or "Unavailable",
                    "Words": record["analytics"]["word_count"],
                    "Sentences": record["analytics"]["sentence_count"],
                    "Major topics": ", ".join(record["analytics"]["major_topics"]),
                }
                for record in analyzed_documents
            ],
            hide_index=True,
            use_container_width=True,
        )

    for index, record in enumerate(documents.values()):
        with st.expander(record["filename"], expanded=not record["text"]):
            if record["error"]:
                st.error(record["error"])
            if record["text"]:
                st.text_area(
                    "Extracted text",
                    value=record["text"],
                    height=260,
                    key=f"text-{index}",
                )
                st.caption(f"Processed data: {record['processed_path']}")
                analytics = record.get("analytics")
                if analytics:
                    st.markdown(
                        f"**Category:** {analytics['category']} "
                        f"({analytics['classification_confidence']}% heuristic confidence)"
                    )
                    summary_tab, details_tab, frequency_tab = st.tabs(
                        ["Summaries", "Topics and keywords", "Term frequency"]
                    )
                    with summary_tab:
                        short_column, detailed_column = st.columns(2)
                        short_column.markdown("**Short summary**")
                        short_column.write(analytics["short_summary"] or "No summary available.")
                        detailed_column.markdown("**Detailed summary**")
                        detailed_column.write(
                            analytics["detailed_summary"] or "No summary available."
                        )
                        st.markdown("**Bullet-point summary**")
                        if analytics["bullet_summary"]:
                            for bullet in analytics["bullet_summary"]:
                                st.markdown(f"- {bullet}")
                        else:
                            st.write("No summary available.")
                    with details_tab:
                        st.write(
                            f"Pages: {analytics['page_count'] or 'Unavailable'} · "
                            f"Words: {analytics['word_count']:,} · "
                            f"Sentences: {analytics['sentence_count']:,}"
                        )
                        st.markdown("**Major topics**")
                        st.write(", ".join(analytics["major_topics"]) or "None detected")
                        st.markdown("**Important keywords (TF-IDF)**")
                        st.write(", ".join(analytics["important_keywords"]) or "None detected")
                    with frequency_tab:
                        st.dataframe(
                            analytics["frequent_terms"],
                            hide_index=True,
                            use_container_width=True,
                        )
                with st.expander(f"View {len(record['chunks'])} processed chunks"):
                    for chunk in record["chunks"]:
                        page_label = (
                            f"page {chunk['page_number']}"
                            if chunk["page_number"] is not None
                            else "page unavailable"
                        )
                        st.markdown(
                            f"**Chunk {chunk['chunk_number']} · {page_label} · "
                            f"{chunk['token_count']} tokens**"
                        )
                        st.write(chunk["text"])
            elif record["stored_name"]:
                st.warning(
                    "No text was extracted. For scanned files, check image clarity and "
                    "confirm that Tesseract OCR is installed."
                )

learning_documents = [
    (fingerprint, record)
    for fingerprint, record in documents.items()
    if record.get("text")
]

st.divider()
st.subheader("Student Learning")
st.caption("Generate simple explanations and source-grounded practice questions.")

if not learning_documents:
    st.info("Upload a document with extracted text to use the learning tools.")
else:
    selected_fingerprint = st.selectbox(
        "Choose a document",
        options=[fingerprint for fingerprint, _ in learning_documents],
        format_func=lambda fingerprint: documents[fingerprint]["filename"],
        key="learning-document-select",
    )
    selected_record = documents[selected_fingerprint]
    selected_document_chunks = selected_record.get("chunks", [])

    if selected_document_chunks:
        default_chunk_indices = list(range(min(2, len(selected_document_chunks))))
        selected_chunk_indices = st.multiselect(
            "Select document content",
            options=list(range(len(selected_document_chunks))),
            default=default_chunk_indices,
            format_func=lambda index: (
                f"Chunk {selected_document_chunks[index]['chunk_number']}"
                + (
                    f" · page {selected_document_chunks[index]['page_number']}"
                    if selected_document_chunks[index]["page_number"] is not None
                    else ""
                )
            ),
            key=f"learning-chunks-{selected_fingerprint}",
        )
        selected_chunks = [
            selected_document_chunks[index] for index in selected_chunk_indices
        ]
        selected_text = "\n\n".join(chunk["text"] for chunk in selected_chunks)
    else:
        selected_chunk_indices = []
        selected_chunks = []
        selected_text = selected_record["text"]

    if selected_chunks:
        source_pages = sorted(
            {
                chunk["page_number"]
                for chunk in selected_chunks
                if chunk["page_number"] is not None
            }
        )
        page_description = (
            f"pages {', '.join(str(page) for page in source_pages)}"
            if source_pages
            else "page unavailable"
        )
        st.caption(
            f"Selected source: {selected_record['filename']} · {page_description} · "
            f"{len(selected_chunks)} chunk(s)"
        )

    learning_state = st.session_state.setdefault("student_learning", {})
    explain_column, quiz_column = st.columns(2)
    with explain_column:
        st.markdown("**Explain Simply**")
        st.caption("Rewrite selected document content in clear, student-friendly language.")
        if st.button(
            "Explain Simply",
            disabled=not selected_text,
            key="explain-simply-button",
        ):
            with st.spinner("Preparing an explanation from the selected text..."):
                try:
                    explanation = explain_simply(selected_text)
                except Exception as error:
                    st.error(f"Could not generate an explanation: {error}")
                else:
                    learning_state["explanation"] = {
                        "fingerprint": selected_fingerprint,
                        "chunk_indices": selected_chunk_indices,
                        "text": explanation,
                    }

        explanation_state = learning_state.get("explanation", {})
        if (
            explanation_state.get("fingerprint") == selected_fingerprint
            and explanation_state.get("chunk_indices") == selected_chunk_indices
        ):
            st.markdown(explanation_state["text"])
            with st.expander("Inspect explanation source"):
                st.caption(f"Source document: {selected_record['filename']}")
                st.text(selected_text)

    with quiz_column:
        st.markdown("**Quiz mode**")
        question_columns = st.columns(3)
        question_count = question_columns[0].selectbox(
            "Questions", options=range(1, 11), index=4, key="learning-question-count"
        )
        difficulty = question_columns[1].selectbox(
            "Difficulty", options=DIFFICULTIES, index=1, key="learning-difficulty"
        )
        question_type = question_columns[2].selectbox(
            "Question type",
            options=QUESTION_TYPES,
            index=QUESTION_TYPES.index("Mixed"),
            key="learning-question-type",
        )

        if st.button(
            "Generate Quiz",
            disabled=not selected_text,
            key="generate-learning-quiz",
        ):
            with st.spinner("Generating questions from the selected document text..."):
                try:
                    generated_questions = generate_quiz(
                        selected_text,
                        number_of_questions=min(question_count, MAX_QUESTIONS),
                        difficulty=difficulty,
                        question_type=question_type,
                    )
                except Exception as error:
                    learning_state["quiz_error"] = str(error)
                    learning_state.pop("quiz", None)
                else:
                    learning_state.pop("quiz_error", None)
                    learning_state["quiz"] = {
                        "fingerprint": selected_fingerprint,
                        "chunk_indices": selected_chunk_indices,
                        "questions": generated_questions,
                        "difficulty": difficulty,
                        "question_type": question_type,
                        "result": None,
                    }

        if learning_state.get("quiz_error"):
            st.error(learning_state["quiz_error"])

    quiz_state = learning_state.get("quiz", {})
    quiz_is_current = (
        quiz_state.get("fingerprint") == selected_fingerprint
        and quiz_state.get("chunk_indices") == selected_chunk_indices
    )
    if quiz_is_current:
        st.markdown(
            f"**{quiz_state['difficulty']} quiz · {len(quiz_state['questions'])} "
            f"question(s) · {quiz_state['question_type']}**"
        )
        with st.form("student-learning-quiz-form"):
            submitted_responses = {}
            for question_index, question_data in enumerate(quiz_state["questions"]):
                st.markdown(
                    f"**{question_index + 1}. {question_data['question']}**"
                )
                if question_data["type"] == "MCQ":
                    submitted_responses[question_index] = st.radio(
                        "Select one answer",
                        options=["Select an option"] + question_data["options"],
                        key=f"quiz-answer-{question_index}",
                        label_visibility="collapsed",
                    )
                elif question_data["type"] == "Short-answer":
                    submitted_responses[question_index] = st.text_input(
                        "Your answer",
                        key=f"quiz-answer-{question_index}",
                    )
                else:
                    submitted_responses[question_index] = st.text_area(
                        "Your answer",
                        key=f"quiz-answer-{question_index}",
                    )
            submitted = st.form_submit_button("Submit quiz")

        if submitted:
            learning_state["quiz"]["result"] = grade_quiz(
                quiz_state["questions"],
                {
                    index: (
                        ""
                        if answer == "Select an option"
                        else answer
                    )
                    for index, answer in submitted_responses.items()
                },
            )

        quiz_result = learning_state["quiz"].get("result")
        if quiz_result:
            st.metric(
                "Quiz score",
                f"{quiz_result['correct']} / {quiz_result['total']} "
                f"({quiz_result['score_percent']}%)",
            )
            for result in quiz_result["results"]:
                if result["correct"]:
                    st.success(f"Correct: {result['question']}")
                else:
                    st.error(f"Review: {result['question']}")
                    st.write(f"Your answer: {result['user_answer']}")
                st.write(f"Answer: {result['correct_answer']}")
                st.write(f"Explanation: {result['explanation']}")
                st.caption(f"Source evidence: {result['evidence']}")
            if quiz_result["weak_topics"]:
                st.warning(
                    "Topics to review: " + ", ".join(quiz_result["weak_topics"])
                )
            else:
                st.success("No weak topics identified from this quiz.")

st.divider()
st.subheader("Document Comparison")
st.caption(
    "Compare two uploaded documents using shared embeddings, topic extraction, and text differences."
)

comparison_documents = [
    (fingerprint, record)
    for fingerprint, record in documents.items()
    if record.get("text")
]
if len(comparison_documents) < 2:
    st.info("Upload at least two documents with extracted text to compare them.")
else:
    first_column, second_column = st.columns(2)
    comparison_ids = [fingerprint for fingerprint, _ in comparison_documents]
    first_id = first_column.selectbox(
        "First document",
        options=comparison_ids,
        format_func=lambda fingerprint: documents[fingerprint]["filename"],
        key="comparison-first-document",
    )
    second_options = [fingerprint for fingerprint in comparison_ids if fingerprint != first_id]
    second_id = second_column.selectbox(
        "Second document",
        options=second_options,
        format_func=lambda fingerprint: documents[fingerprint]["filename"],
        key="comparison-second-document",
    )

    comparison_state = st.session_state.setdefault("document_comparison", {})
    if st.button("Compare documents", key="compare-documents-button"):
        first_record = documents[first_id]
        second_record = documents[second_id]
        first_analytics = first_record.get("analytics") or {}
        second_analytics = second_record.get("analytics") or {}
        with st.spinner("Comparing semantic content and document changes..."):
            try:
                result = compare_documents(
                    first_record["filename"],
                    first_record["text"],
                    second_record["filename"],
                    second_record["text"],
                    first_topics=first_analytics.get("major_topics"),
                    second_topics=second_analytics.get("major_topics"),
                )
            except Exception as error:
                comparison_state["error"] = str(error)
                comparison_state.pop("result", None)
            else:
                comparison_state.pop("error", None)
                comparison_state["first_id"] = first_id
                comparison_state["second_id"] = second_id
                comparison_state["result"] = result

    if comparison_state.get("error"):
        st.error(f"Could not compare documents: {comparison_state['error']}")

    comparison = comparison_state.get("result")
    if (
        comparison
        and comparison_state.get("first_id") == first_id
        and comparison_state.get("second_id") == second_id
    ):
        st.metric("Semantic similarity", f"{comparison['similarity_score']}%")
        common_column, unique_column = st.columns(2)
        common_column.markdown("**Common topics**")
        common_column.write(", ".join(comparison["common_topics"]) or "No shared topics detected.")
        unique_column.markdown("**Unique topics**")
        unique_column.write(
            f"{comparison['first_document']}: "
            f"{', '.join(comparison['first_unique_topics']) or 'None detected'}"
        )
        unique_column.write(
            f"{comparison['second_document']}: "
            f"{', '.join(comparison['second_unique_topics']) or 'None detected'}"
        )

        st.markdown("**Changed values, dates, and rules**")
        value_column_a, value_column_b = st.columns(2)
        value_column_a.write(
            f"Only in {comparison['first_document']}: "
            f"{', '.join(comparison['first_only_values']) or 'None detected'}"
        )
        value_column_b.write(
            f"Only in {comparison['second_document']}: "
            f"{', '.join(comparison['second_only_values']) or 'None detected'}"
        )

        st.markdown("**Important text differences**")
        if comparison["sentence_changes"]:
            st.dataframe(
                comparison["sentence_changes"],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No sentence-level text differences detected.")

st.divider()
st.subheader("Ask your documents")
st.caption("Answers are generated only from retrieved document chunks.")

answer_language_preference = st.selectbox(
    "Answer language",
    options=ANSWER_LANGUAGE_OPTIONS,
    key="answer-language-preference",
)
speak_answers = st.checkbox("Read answers aloud", value=False, key="speak-answers")

with st.expander("Voice input (optional)"):
    voice_input_language = st.selectbox(
        "Spoken question language",
        options=("English", "Tamil"),
        key="voice-input-language",
    )
    voice_locale = "ta-IN" if voice_input_language == "Tamil" else "en-US"
    try:
        recognized_question = speech_to_text(
            language=voice_locale,
            start_prompt="Start recording",
            stop_prompt="Stop and transcribe",
            just_once=True,
            use_container_width=True,
            key="document-question-voice-input",
        )
    except Exception as error:
        recognized_question = None
        st.error(f"Speech recognition failed: {error}")

    if recognized_question:
        st.session_state["recognized-voice-question"] = recognized_question
    voice_question = st.text_input(
        "Recognized question (edit before sending if needed)",
        key="recognized-voice-question",
    )
    send_voice_question = st.button(
        "Ask this voice question",
        disabled=not voice_question.strip(),
        key="send-voice-question",
    )
    st.caption(
        "Speech recognition uses your browser's configured service; allow microphone "
        "access and internet connectivity."
    )

chat_history = st.session_state.setdefault("chat_history", [])
for message in chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "user" and message.get("detected_language"):
            st.caption(
                f"Detected: {message['detected_language']} · "
                f"Answer: {message.get('response_language', 'English')}"
            )
        if message["role"] == "assistant":
            if message.get("interpreted_question"):
                st.caption(f"Follow-up interpreted as: {message['interpreted_question']}")
            if message.get("confidence"):
                _render_confidence(message["confidence"])
            _render_sources(message.get("sources", []), message.get("source_note", ""))

typed_question = st.chat_input("Ask a question about your uploaded documents")
question = voice_question if send_voice_question else typed_question
if question:
    detected_language, response_language = select_answer_language(
        question, answer_language_preference
    )
    prior_history = list(chat_history)
    chat_history.append(
        {
            "role": "user",
            "content": question,
            "detected_language": detected_language,
            "response_language": response_language,
        }
    )
    with st.chat_message("user"):
        st.markdown(question)
        st.caption(f"Detected: {detected_language} · Answer: {response_language}")

    resolved_question = question
    contextualization_note = ""
    try:
        resolved_question = resolve_followup_question(question, prior_history)
    except Exception:
        contextualization_note = (
            "Follow-up context could not be resolved; searched the original question."
        )

    sources = []
    confidence = {"score": 0, "label": "Unavailable", "best_similarity": 0.0}
    source_note = ""
    error_detail = ""
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                if detected_language == "Tamil":
                    try:
                        english_retrieval_question = translate_question_for_retrieval(
                            resolved_question
                        )
                    except Exception:
                        english_retrieval_question = ""

                    if english_retrieval_question:
                        sources = retrieve_relevant_chunks(english_retrieval_question)
                        confidence = assess_retrieval_confidence(sources)
                    else:
                        sources = []
                        confidence = {"score": 0, "label": "No match", "best_similarity": 0.0}

                    if not sources or confidence["label"] == "Low":
                        tamil_sources = retrieve_relevant_chunks(
                            resolved_question,
                            max_distance=0.65,
                        )
                        tamil_confidence = assess_retrieval_confidence(
                            tamil_sources,
                            cross_lingual=True,
                        )
                        if tamil_confidence["label"] not in {"No match", "Low"}:
                            sources = tamil_sources
                            confidence = tamil_confidence
                else:
                    sources = retrieve_relevant_chunks(resolved_question)
                    confidence = assess_retrieval_confidence(sources)
            except Exception as error:
                answer = "I could not answer because document retrieval failed."
                confidence = {"score": 0, "label": "Unavailable", "best_similarity": 0.0}
                error_detail = str(error)
            else:
                if not sources or confidence["label"] == "Low":
                    answer = NO_ANSWER
                    source_note = (
                        "These low-confidence matches were not used to generate an answer."
                        if sources
                        else "No relevant source chunks passed the retrieval threshold."
                    )
                else:
                    try:
                        answer = generate_answer(
                            question,
                            sources,
                            contextualized_question=resolved_question,
                            response_language=response_language,
                        )
                    except Exception as error:
                        answer = "I could not answer because the local answer model failed."
                        source_note = "Retrieved source chunks are shown for inspection."
                        error_detail = str(error)
                    else:
                        if answer == NO_ANSWER:
                            source_note = (
                                "The answer model did not find enough support in these excerpts."
                            )

        if contextualization_note:
            st.caption(contextualization_note)
        if resolved_question != question:
            st.caption(f"Follow-up interpreted as: {resolved_question}")
        st.markdown(answer)
        if error_detail:
            st.caption(error_detail)
        _render_confidence(confidence)
        _render_sources(sources, source_note)
        if speak_answers:
            try:
                speak_answer(answer, response_language)
            except Exception as error:
                st.caption(f"Text-to-speech is unavailable: {error}")

    chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence": confidence,
            "interpreted_question": (
                resolved_question if resolved_question != question else ""
            ),
            "source_note": source_note,
        }
    )