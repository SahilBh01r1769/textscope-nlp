import nlp_core as core


SAMPLE = (
    "Northstar Labs launched Atlas in Bengaluru on Tuesday. "
    "The assistant summarizes manuals and extracts part numbers for engineers. "
    "Pilot teams liked the fast lookup, but two teams reported missed component names. "
    "Northstar plans a broader rollout in October after publishing an evaluation report."
)


def test_document_profile_has_interpretable_metrics():
    profile = core.document_profile(SAMPLE)
    assert profile["words"] > 20
    assert profile["sentences"] == 4
    assert 0 < profile["lexical_diversity"] <= 1
    assert profile["avg_sentence_words"] > 0


def test_sentiment_uses_intensity_not_fake_confidence():
    result = core.analyse_sentiment("I love the fast support. The crashes are frustrating.")
    assert "intensity" in result
    assert "confidence" not in result
    assert len(result["sentences"]) == 2


def test_keywords_keep_sentence_evidence():
    result = core.extract_keywords(SAMPLE, top_n=8)
    assert result["keywords"]
    assert all(row["sentence_ids"] for row in result["keywords"])
    assert all(row["score"] > 0 for row in result["keywords"])


def test_extractive_summary_returns_coverage_diagnostics():
    result = core.summarize_text(SAMPLE, num_sentences=2, mode="extractive")
    assert result["mode"] == "extractive"
    assert result["summary"]
    assert result["summary_words"] < result["original_words"]
    assert 0 <= result["keyword_coverage"] <= 100
    assert 0 <= result["entity_retention"] <= 100


def test_qa_returns_supporting_sentence(monkeypatch):
    class FakeQA:
        def __call__(self, *, question, context):
            answer = "Bengaluru"
            start = context.index(answer)
            return {"answer": answer, "score": 0.91, "start": start, "end": start + len(answer)}

    monkeypatch.setattr(core, "get_qa", lambda: FakeQA())
    result = core.answer_question(SAMPLE, "Where was Atlas launched?", threshold=0.12)
    assert result["answer_found"] is True
    assert result["answer"] == "Bengaluru"
    assert result["evidence_sentence_id"] == 1
    assert "Bengaluru" in result["evidence"]


def test_qa_abstains_when_span_is_weak(monkeypatch):
    class FakeQA:
        def __call__(self, *, question, context):
            return {"answer": "October", "score": 0.03, "start": 0, "end": 7}

    monkeypatch.setattr(core, "get_qa", lambda: FakeQA())
    result = core.answer_question(SAMPLE, "What is the launch price?", threshold=0.12)
    assert result["answer_found"] is False
    assert result["evidence"] is None
    assert "No sufficiently supported answer" in result["answer"]
