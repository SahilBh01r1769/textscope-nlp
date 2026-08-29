# TextScope — Evidence-Aware NLP Workbench

> **Understand a document, compress it, and verify answers without hiding the source evidence.**

TextScope is an NLP engineering project that turns a single English document into a connected set of interpretable views: document structure, sentiment trajectory, named entities, salient concepts, summaries, extractive question answering, and dependency syntax.

The project is deliberately built around **traceability** rather than a collection of disconnected model demos. Extracted concepts retain source sentence IDs, QA returns the supporting sentence (or abstains), and summaries include simple coverage diagnostics so generated text is not presented as automatically faithful.

## Why this project exists

Many beginner NLP applications expose several tasks in separate tabs but do not explain how their outputs relate to the source text. TextScope treats the source document as the common evidence layer.

```text
Document
   │
   ├── linguistic structure ──► profile + syntax
   ├── lexical tone ──────────► overall + sentence trajectory
   ├── entities/concepts ─────► source sentence references
   ├── summarization ─────────► compression + coverage diagnostics
   └── extractive QA ─────────► answer + supporting sentence / abstention
```

## Project highlights

| Capability | Implementation |
|---|---|
| Document fingerprint | word/sentence counts, reading time, sentence length, lexical diversity, entity density |
| Sentiment | TextBlob polarity/subjectivity plus sentence-level trajectory and variation |
| Named entities | spaCy NER with source sentence IDs |
| Key concepts | noun-phrase/token candidates with sentence-aware TF-IDF-style scoring |
| Extractive summary | evidence-ranked source sentences |
| Abstractive summary | chunk-aware `facebook/bart-large-cnn` |
| Summary diagnostics | compression, source-entity retention and top-keyword coverage |
| Question answering | `deepset/roberta-base-squad2` with evidence sentence mapping and configurable abstention threshold |
| Syntax | spaCy dependency parsing |
| Interfaces | Streamlit workbench + optional Flask JSON API |
| Quality | shared NLP core, lazy transformer loading, unit tests and GitHub Actions |

## The important design choices

### 1. One source of NLP behavior

`nlp_core.py` contains the actual analysis logic. The Streamlit app and Flask API both call the same functions instead of maintaining two implementations that can drift apart.

### 2. Evidence is preserved

Sentence IDs are attached to named entities and key concepts. Extractive QA maps the selected answer span back to its containing source sentence.

That turns an answer from:

```text
Bengaluru
```

into something inspectable:

```text
Answer: Bengaluru
Evidence: S1 — “Northstar Labs launched Atlas in Bengaluru on Tuesday.”
```

### 3. QA can abstain

Extractive QA models always have a “best” span, even when the document does not really answer the question. TextScope applies a configurable evidence threshold and returns no supported answer when the span is too weak or cannot be mapped cleanly to the source.

The model score is still not treated as a guarantee of factual correctness.

### 4. Sentiment intensity is not called confidence

TextBlob provides polarity and subjectivity, not a calibrated class probability. TextScope therefore reports the magnitude of polarity as **intensity** instead of relabelling it as model confidence.

### 5. Summaries expose coverage diagnostics

For each summary TextScope reports:

- compression rate;
- source named-entity retention;
- top-keyword coverage.

These are useful warning signals for omitted information, but they are explicitly **not factuality metrics**.

### 6. Transformer models are lazy-loaded

The fast document analysis path uses spaCy and TextBlob only. BART and RoBERTa are downloaded/loaded only when abstractive summarization or QA is requested, keeping initial app startup lighter.

## Streamlit walkthrough

The interface follows three questions rather than six unrelated model tabs:

### Understand

Inspect the document fingerprint, sentiment trajectory, named entities, concepts and sentence-level evidence.

### Compress

Generate an extractive or abstractive summary and inspect what source information survived compression.

### Verify

Ask a question and inspect the exact source sentence supporting the extracted answer. Weakly supported answers are rejected.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first call to abstractive summarization or QA downloads the corresponding Hugging Face model.

### Optional Flask API

```bash
python app.py
```

Example routes:

```text
GET  /
POST /api/document
POST /api/sentiment
POST /api/summarize
POST /api/ner
POST /api/keywords
POST /api/qa
POST /api/dependency
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests exercise the fast pipeline without downloading transformer checkpoints. QA behavior is tested with a deterministic fake model so CI verifies evidence mapping and abstention logic without network/model variability.

## Repository structure

```text
.
├── app.py                  # Optional Flask API
├── nlp_core.py             # Shared NLP/evidence pipeline
├── streamlit_app.py        # TextScope UI
├── requirements.txt
├── requirements-dev.txt
├── tests/
│   └── test_nlp_core.py
└── .github/workflows/
    ├── tests.yml
    └── demo-smoke.yml
```

## Models and attribution

TextScope uses established pretrained components rather than claiming custom model training:

- spaCy `en_core_web_sm` for linguistic annotation and NER;
- TextBlob for lexical sentiment baseline;
- Facebook BART Large CNN for optional abstractive summarization;
- Deepset RoBERTa Base SQuAD2 for optional extractive QA.

The engineering contribution is the **evidence-aware orchestration, diagnostics, shared architecture and interactive analysis workflow** around these components.

## Limitations

- English-focused pipeline.
- TextBlob sentiment is a lexical baseline and can struggle with sarcasm, context and domain-specific language.
- spaCy small-model NER is lightweight rather than state of the art.
- entity retention and keyword coverage do not prove summary factuality.
- extractive QA can still select misleading spans above the threshold.
- transformer inference on free CPU hosting can be slow on first use.
- very large documents would benefit from retrieval/chunk selection before QA.

## Next experiments

- sentence retrieval before QA for long documents;
- semantic keyword clustering with sentence embeddings;
- contradiction/entailment checks between summaries and source evidence;
- benchmark QA abstention on answerable/unanswerable examples;
- evaluate summary coverage alongside ROUGE/BERTScore on a reproducible dataset;
- multilingual model adapters while preserving the same evidence interface.

---

**Portfolio framing:** TextScope is a document-intelligence and NLP engineering workbench, not a claim of training new foundation models. Its focus is making pretrained NLP components easier to inspect, compare and trust responsibly.
