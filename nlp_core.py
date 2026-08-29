"""Core NLP pipeline for TextScope.

The module intentionally mixes lightweight linguistic analysis with transformer
models. Heavy Hugging Face models are lazy-loaded; the fast document analysis
path only needs spaCy and TextBlob.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import re
from typing import Iterable

import spacy
from textblob import TextBlob

SUMMARIZER_MODEL = "facebook/bart-large-cnn"
QA_MODEL = "deepset/roberta-base-squad2"


@lru_cache(maxsize=1)
def get_nlp():
    """Load the packaged spaCy model once."""
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def get_summarizer():
    from transformers import pipeline

    return pipeline("summarization", model=SUMMARIZER_MODEL, device=-1)


@lru_cache(maxsize=1)
def get_qa():
    from transformers import pipeline

    return pipeline("question-answering", model=QA_MODEL, device=-1)


def _label_sentiment(polarity: float) -> str:
    if polarity > 0.1:
        return "Positive"
    if polarity < -0.1:
        return "Negative"
    return "Neutral"


def _sentences(doc) -> list:
    return [sent for sent in doc.sents if sent.text.strip()]


def _content_tokens(doc) -> list:
    return [
        token
        for token in doc
        if not token.is_space and not token.is_punct and not token.is_stop
    ]


def sentence_records(text: str) -> list[dict]:
    doc = get_nlp()(text)
    records = []
    for index, sent in enumerate(_sentences(doc), start=1):
        polarity = round(TextBlob(sent.text).sentiment.polarity, 4)
        records.append(
            {
                "id": index,
                "text": sent.text.strip(),
                "start": sent.start_char,
                "end": sent.end_char,
                "polarity": polarity,
                "label": _label_sentiment(polarity),
            }
        )
    return records


def document_profile(text: str) -> dict:
    doc = get_nlp()(text)
    sentences = _sentences(doc)
    words = [token for token in doc if token.is_alpha]
    content = _content_tokens(doc)
    lemmas = [token.lemma_.lower() for token in content if token.lemma_.strip()]
    entities = list(doc.ents)

    word_count = len(words)
    sentence_count = len(sentences)
    unique_content = len(set(lemmas))
    lexical_diversity = round(unique_content / max(len(lemmas), 1), 3)
    avg_sentence_words = round(word_count / max(sentence_count, 1), 1)

    return {
        "words": word_count,
        "sentences": sentence_count,
        "characters": len(text),
        "reading_minutes": round(word_count / 220, 1),
        "avg_sentence_words": avg_sentence_words,
        "lexical_diversity": lexical_diversity,
        "entities": len(entities),
        "entity_density": round((len(entities) / max(word_count, 1)) * 100, 2),
    }


def analyse_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)
    records = sentence_records(text)
    values = [row["polarity"] for row in records]
    mean = sum(values) / max(len(values), 1)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)

    return {
        "label": _label_sentiment(polarity),
        "polarity": polarity,
        "subjectivity": subjectivity,
        # This is deliberately called intensity, not confidence. TextBlob does
        # not return a calibrated confidence probability.
        "intensity": round(abs(polarity) * 100, 1),
        "sentence_variation": round(math.sqrt(variance), 3),
        "sentences": records,
    }


def run_ner(text: str) -> dict:
    doc = get_nlp()(text)
    entities = []
    for ent in doc.ents:
        sentence_id = next(
            (index for index, sent in enumerate(_sentences(doc), 1) if sent.start_char <= ent.start_char < sent.end_char),
            None,
        )
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "desc": spacy.explain(ent.label_) or ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "sentence_id": sentence_id,
            }
        )

    counts = Counter(entity["label"] for entity in entities)
    return {
        "entities": entities,
        "breakdown": [{"label": key, "count": value} for key, value in counts.most_common()],
        "total": len(entities),
    }


def _candidate_terms(doc) -> dict[str, set[int]]:
    """Return candidate terms and the sentence ids in which they occur."""
    candidates: dict[str, set[int]] = {}
    sentences = _sentences(doc)

    for sid, sent in enumerate(sentences, start=1):
        sent_doc = sent.as_doc()
        for chunk in sent_doc.noun_chunks:
            parts = [
                token.lemma_.lower()
                for token in chunk
                if token.is_alpha and not token.is_stop and len(token.text) > 2
            ]
            if parts:
                phrase = " ".join(parts)
                if len(phrase) > 2:
                    candidates.setdefault(phrase, set()).add(sid)

        for token in sent:
            if (
                token.is_alpha
                and token.pos_ in {"NOUN", "PROPN", "ADJ"}
                and not token.is_stop
                and len(token.text) > 2
            ):
                candidates.setdefault(token.lemma_.lower(), set()).add(sid)

    return candidates


def extract_keywords(text: str, top_n: int = 12) -> dict:
    doc = get_nlp()(text)
    sentences = _sentences(doc)
    sentence_count = max(len(sentences), 1)
    candidates = _candidate_terms(doc)

    if not candidates:
        return {"keywords": [], "total_words": document_profile(text)["words"]}

    normalized_text = " ".join(token.lemma_.lower() for token in doc if token.is_alpha)
    scored = []
    for term, sentence_ids in candidates.items():
        # Phrase-aware term frequency. Word-boundary matching avoids substring
        # inflation (e.g. "art" inside "article").
        frequency = len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", normalized_text))
        frequency = max(frequency, 1)
        tf = frequency / max(len(normalized_text.split()), 1)
        idf = math.log((sentence_count + 1) / (len(sentence_ids) + 1)) + 1
        phrase_bonus = 1 + 0.12 * max(len(term.split()) - 1, 0)
        score = tf * idf * phrase_bonus
        scored.append(
            {
                "term": term,
                "score": round(score, 5),
                "count": frequency,
                "sentence_ids": sorted(sentence_ids),
            }
        )

    scored.sort(key=lambda row: (row["score"], row["count"]), reverse=True)
    return {
        "keywords": scored[: max(1, top_n)],
        "total_words": document_profile(text)["words"],
    }


def _extractive_summary(text: str, num_sentences: int) -> str:
    doc = get_nlp()(text)
    sentences = _sentences(doc)
    if not sentences:
        return ""

    keywords = extract_keywords(text, top_n=20)["keywords"]
    weights = {row["term"]: row["score"] for row in keywords}
    scored = []
    for idx, sent in enumerate(sentences):
        lemma_text = " ".join(token.lemma_.lower() for token in sent if token.is_alpha)
        score = sum(weight for term, weight in weights.items() if term in lemma_text)
        # Small position prior: introductions/conclusions often carry context.
        position_bonus = 1.08 if idx in {0, len(sentences) - 1} else 1.0
        scored.append((score * position_bonus, idx, sent.text.strip()))

    selected = sorted(scored, reverse=True)[: min(num_sentences, len(sentences))]
    selected.sort(key=lambda item: item[1])
    return " ".join(item[2] for item in selected)


def _chunk_sentences(text: str, max_words: int = 650) -> list[str]:
    doc = get_nlp()(text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in _sentences(doc):
        sentence_words = len(sent.text.split())
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sent.text.strip())
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def _abstractive_summary(text: str) -> str:
    summarizer = get_summarizer()
    chunks = _chunk_sentences(text)
    partials = []
    for chunk in chunks:
        words = len(chunk.split())
        max_len = min(160, max(55, words // 3))
        min_len = min(max_len - 10, max(25, max_len // 3))
        result = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        partials.append(result[0]["summary_text"].strip())

    combined = " ".join(partials)
    if len(partials) > 1 and len(combined.split()) > 220:
        result = summarizer(combined, max_length=170, min_length=55, do_sample=False)
        return result[0]["summary_text"].strip()
    return combined


def _coverage_diagnostics(source: str, summary: str) -> dict:
    source_entities = {ent.text.lower() for ent in get_nlp()(source).ents}
    summary_entities = {ent.text.lower() for ent in get_nlp()(summary).ents}
    entity_retention = (
        len(source_entities & summary_entities) / len(source_entities) if source_entities else 1.0
    )

    keywords = extract_keywords(source, top_n=10)["keywords"]
    summary_lower = summary.lower()
    covered = [row["term"] for row in keywords if row["term"] in summary_lower]
    keyword_coverage = len(covered) / max(len(keywords), 1)

    return {
        "entity_retention": round(entity_retention * 100, 1),
        "keyword_coverage": round(keyword_coverage * 100, 1),
        "covered_keywords": covered,
    }


def summarize_text(text: str, num_sentences: int = 3, mode: str = "auto") -> dict:
    word_count = len(text.split())
    if word_count == 0:
        raise ValueError("Text is empty.")

    if mode not in {"auto", "extractive", "abstractive"}:
        raise ValueError("mode must be auto, extractive, or abstractive")

    selected_mode = mode
    if mode == "auto":
        selected_mode = "extractive" if word_count < 140 else "abstractive"

    if selected_mode == "extractive":
        summary = _extractive_summary(text, max(1, num_sentences))
        method = "Evidence-ranked extractive summary"
    else:
        summary = _abstractive_summary(text)
        method = "Chunk-aware BART abstractive summary"

    summary_words = len(summary.split())
    diagnostics = _coverage_diagnostics(text, summary)
    return {
        "summary": summary,
        "method": method,
        "mode": selected_mode,
        "original_words": word_count,
        "summary_words": summary_words,
        "compression_rate": round((1 - summary_words / max(word_count, 1)) * 100, 1),
        **diagnostics,
    }


def _find_evidence_sentence(text: str, start: int, end: int) -> tuple[int | None, str | None]:
    doc = get_nlp()(text)
    for sid, sent in enumerate(_sentences(doc), start=1):
        if sent.start_char <= start < sent.end_char or sent.start_char < end <= sent.end_char:
            return sid, sent.text.strip()
    return None, None


def answer_question(text: str, question: str, threshold: float = 0.12) -> dict:
    result = get_qa()(question=question, context=text)
    score = float(result.get("score", 0.0))
    answer = str(result.get("answer", "")).strip()
    start = int(result.get("start", 0))
    end = int(result.get("end", start))
    evidence_id, evidence = _find_evidence_sentence(text, start, end)

    supported = bool(answer) and score >= threshold and evidence is not None
    if not supported:
        answer = "No sufficiently supported answer was found in the supplied text."
        evidence = None
        evidence_id = None

    return {
        "answer": answer,
        "answer_found": supported,
        "score": round(score, 4),
        "confidence": round(score * 100, 1),
        "question": question,
        "evidence": evidence,
        "evidence_sentence_id": evidence_id,
        "span": {"start": start, "end": end} if supported else None,
        "threshold": threshold,
    }


def dependency_parse(text: str) -> dict:
    from spacy import displacy

    doc = get_nlp()(text)
    sentences = _sentences(doc)
    if not sentences:
        raise ValueError("No sentence found.")
    first_sent = sentences[0]
    sent_doc = first_sent.as_doc()
    svg = displacy.render(
        sent_doc,
        style="dep",
        page=False,
        minify=True,
        options={"compact": True, "bg": "#0b1220", "color": "#dbeafe", "font": "Inter"},
    )
    return {
        "svg": svg,
        "sentence": first_sent.text.strip(),
        "tokens": [
            {"text": token.text, "pos": token.pos_, "dep": token.dep_, "head": token.head.text}
            for token in first_sent
        ],
    }


def analyse_document(text: str, keyword_count: int = 12) -> dict:
    """Fast, transformer-free document intelligence pass."""
    return {
        "profile": document_profile(text),
        "sentiment": analyse_sentiment(text),
        "entities": run_ner(text),
        "keywords": extract_keywords(text, top_n=keyword_count),
        "sentences": sentence_records(text),
    }
