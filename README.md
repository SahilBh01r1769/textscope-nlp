# TextScope

TextScope is a small NLP workbench for exploring an English document from a few different angles.

It can show basic document statistics, sentiment, named entities, keywords, summaries, question answering and a dependency parse. The project mostly combines existing NLP libraries and pretrained models behind one Streamlit interface.

## What it uses

- **spaCy** for tokenization, named entities and dependency parsing
- **TextBlob** for simple sentiment analysis
- **BART (`facebook/bart-large-cnn`)** for abstractive summaries
- **RoBERTa (`deepset/roberta-base-squad2`)** for extractive question answering
- a small custom keyword scorer and extractive summarizer

The transformer models are loaded only when summarization or question answering is requested, so the basic document analysis can run without loading them first.

## Main features

Given a document, TextScope can:

- count words and sentences and calculate a few simple text statistics;
- show overall and sentence-level sentiment;
- extract named entities with spaCy;
- rank keywords and short noun phrases;
- create either an extractive or BART-based summary;
- answer a question from the supplied text using an extractive QA model;
- show the source sentence containing the QA answer;
- render a dependency parse for a sentence.

The QA view uses a score threshold and returns no supported answer when the model result is too weak. This is a simple guard around the pretrained model, not a guarantee that accepted answers are correct.

## How the code is organised

Most NLP behaviour lives in `nlp_core.py`. Both the Streamlit app and the optional Flask API call the same functions rather than implementing the tasks separately.

```text
Document
  ├─ spaCy / TextBlob → profile, entities, sentiment, syntax
  ├─ custom scoring   → keywords and extractive summary
  ├─ BART             → abstractive summary
  └─ RoBERTa          → extractive question answering
```

For summaries, the interface also shows basic information such as compression rate and whether common source entities or keywords still appear in the summary. These are only rough checks and should not be treated as factuality scores.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first request for abstractive summarization or QA downloads the corresponding Hugging Face model.

An optional Flask API is also included:

```bash
python app.py
```

## Repository layout

```text
nlp_core.py          NLP functions used by both interfaces
streamlit_app.py     main interactive application
app.py               optional Flask API
tests/               unit tests
requirements.txt     runtime dependencies
```

## Scope

This is not a custom language-model project. Most of the heavier NLP tasks use pretrained models, while the project work is in combining them into one application, keeping the interfaces consistent, and adding a few small utilities around the model outputs.
