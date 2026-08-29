# Hosted TextScope deployment

Recommended Streamlit Community Cloud configuration:

```text
Repository: SahilBh01r1769/NLP_WEB_APP_h
Branch: demo/hosted-textscope
Main file path: streamlit_app.py
Python: 3.11
```

The root `requirements.txt` is deployment-complete because the Streamlit entrypoint is also at repository root.

## Runtime behavior

- Initial document analysis uses spaCy + TextBlob and does not download a transformer checkpoint.
- Abstractive summarization lazy-loads `facebook/bart-large-cnn` on first use.
- Document Q&A lazy-loads `deepset/roberta-base-squad2` on first use.
- First transformer use can therefore be slower on CPU hosting.

The UI labels pretrained-model usage explicitly and does not present these checkpoints as custom-trained models.
