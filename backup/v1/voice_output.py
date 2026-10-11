"""Optional browser-based speech synthesis for generated answers."""

import json

import streamlit.components.v1 as components


def speak_answer(text: str, language: str) -> None:
    """Ask the browser to speak a new answer using an available system voice."""
    if not text.strip():
        return

    locale = "ta-IN" if language == "Tamil" else "en-US"
    safe_text = json.dumps(text).replace("</", "<\\/")
    html = f"""
    <script>
    const answer = {safe_text};
    const utterance = new SpeechSynthesisUtterance(answer);
    utterance.lang = "{locale}";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    </script>
    """
    components.html(html, height=0, scrolling=False)