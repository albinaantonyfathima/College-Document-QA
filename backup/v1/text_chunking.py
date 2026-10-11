"""Create token-bounded, page-aware document chunks for later retrieval."""

from text_preprocessing import normalize_text, split_sentences, tokenize


def chunk_pages(
    pages: list[dict],
    document_name: str,
    max_tokens: int = 250,
    overlap_tokens: int = 40,
) -> list[dict]:
    """Chunk page text and attach source metadata to every chunk."""
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Require max_tokens > 0 and 0 <= overlap_tokens < max_tokens.")

    chunks = []
    for page in pages:
        page_text = normalize_text(page.get("text", ""))
        current_words = []
        current_token_count = 0

        def save_current(carry_overlap: bool) -> None:
            nonlocal current_words, current_token_count
            if not current_words:
                return

            chunk_text = " ".join(current_words)
            tokens = tokenize(chunk_text)
            chunks.append(
                {
                    "document_name": document_name,
                    "page_number": page.get("page_number"),
                    "chunk_number": len(chunks) + 1,
                    "text": chunk_text,
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "sentence_count": len(split_sentences(chunk_text)),
                }
            )

            overlap_words = []
            overlap_count = 0
            if carry_overlap and overlap_tokens:
                for word in reversed(current_words):
                    word_count = len(tokenize(word))
                    if overlap_count + word_count > overlap_tokens:
                        break
                    overlap_words.append(word)
                    overlap_count += word_count
                    if overlap_count >= overlap_tokens:
                        break

            current_words = list(reversed(overlap_words))
            current_token_count = overlap_count

        for sentence in split_sentences(page_text):
            for word in sentence.split():
                word_token_count = len(tokenize(word))
                if current_words and current_token_count + word_token_count > max_tokens:
                    save_current(carry_overlap=True)
                current_words.append(word)
                current_token_count += word_token_count

        save_current(carry_overlap=False)

    return chunks