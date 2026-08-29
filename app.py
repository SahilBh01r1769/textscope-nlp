"""Optional Flask API for TextScope.

The Streamlit demo calls nlp_core directly. This API exists for integration and
reuses the same core functions so there is one source of NLP behavior.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

import nlp_core as core

app = Flask(__name__)
CORS(app)


def _payload():
    return request.get_json(silent=True) or {}


def _text(min_words: int = 5):
    text = str(_payload().get("text") or "").strip()
    if not text:
        return None, (jsonify({"error": "No text provided."}), 400)
    if len(text.split()) < min_words:
        return None, (jsonify({"error": f"Please provide at least {min_words} words."}), 400)
    return text, None


@app.get("/")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "TextScope NLP API",
            "features": ["document", "sentiment", "summary", "ner", "keywords", "qa", "dependency"],
        }
    )


@app.post("/api/document")
def document():
    data = _payload()
    text, error = _text(10)
    if error:
        return error
    return jsonify(core.analyse_document(text, keyword_count=int(data.get("top_n", 12))))


@app.post("/api/sentiment")
def sentiment():
    text, error = _text(3)
    return error if error else jsonify(core.analyse_sentiment(text))


@app.post("/api/summarize")
def summarize():
    data = _payload()
    text, error = _text(30)
    if error:
        return error
    try:
        result = core.summarize_text(
            text,
            num_sentences=int(data.get("sentences", 3)),
            mode=str(data.get("mode", "auto")),
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/ner")
def ner():
    text, error = _text(5)
    return error if error else jsonify(core.run_ner(text))


@app.post("/api/keywords")
def keywords():
    data = _payload()
    text, error = _text(10)
    if error:
        return error
    return jsonify(core.extract_keywords(text, top_n=int(data.get("top_n", 12))))


@app.post("/api/qa")
def qa():
    data = _payload()
    text = str(data.get("text") or "").strip()
    question = str(data.get("question") or "").strip()
    if len(text.split()) < 10:
        return jsonify({"error": "Provide at least 10 words of context."}), 400
    if not question:
        return jsonify({"error": "No question provided."}), 400
    threshold = float(data.get("threshold", 0.12))
    return jsonify(core.answer_question(text, question, threshold=threshold))


@app.post("/api/dependency")
def dependency():
    text, error = _text(3)
    return error if error else jsonify(core.dependency_parse(text))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
