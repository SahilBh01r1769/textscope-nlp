"""TextScope — evidence-aware NLP workbench."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import nlp_core as core

st.set_page_config(
    page_title="TextScope · Evidence-Aware NLP",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{--ink:#e8eef8;--muted:#8b9bb4;--panel:#0c1422;--line:rgba(148,163,184,.14);--accent:#67e8f9;--violet:#a78bfa;}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 18% 0%,#10243a 0,#07101c 30%,#050b13 68%);color:var(--ink);}
[data-testid="stSidebar"]{background:#07101a;border-right:1px solid var(--line);}
h1,h2,h3{letter-spacing:-.025em;color:var(--ink)!important;}
.hero{padding:1.2rem 0 .6rem}.eyebrow{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:#67e8f9;font-weight:700}
.hero-title{font-size:3rem;line-height:1.02;font-weight:850;margin:.35rem 0;background:linear-gradient(90deg,#e0f2fe,#67e8f9 42%,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-copy{max-width:820px;color:#9fb0c8;font-size:1.02rem;line-height:1.65}
.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0 22px}.flow-card,.panel{background:rgba(12,20,34,.78);border:1px solid var(--line);border-radius:16px;padding:16px}.flow-card b{display:block;color:#e8eef8;margin-bottom:5px}.flow-card span{color:#8192aa;font-size:.86rem;line-height:1.45}
.insight{border-left:3px solid #67e8f9;background:rgba(103,232,249,.055);padding:13px 16px;border-radius:0 12px 12px 0;color:#cbd9ea}.evidence{border-left:3px solid #a78bfa;background:rgba(167,139,250,.06);padding:13px 16px;border-radius:0 12px 12px 0;color:#d9d2f7}
.pill{display:inline-block;border:1px solid rgba(103,232,249,.24);background:rgba(103,232,249,.06);color:#b9f6ff;border-radius:999px;padding:5px 10px;margin:3px;font-size:.8rem}.sent-id{color:#67e8f9;font-size:.72rem;font-weight:700;margin-right:6px}.small{color:#7f91aa;font-size:.84rem;line-height:1.5}
[data-testid="stMetric"]{background:rgba(12,20,34,.76);border:1px solid var(--line);padding:13px;border-radius:14px}[data-testid="stMetricLabel"]{color:#8fa2bb}
.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{height:42px;border-radius:9px;padding:0 15px;background:#0b1421;color:#8496ae}.stTabs [aria-selected="true"]{background:rgba(103,232,249,.11)!important;color:#bff7ff!important}
.stButton>button{border-radius:10px;border:1px solid rgba(103,232,249,.26);background:rgba(103,232,249,.08);color:#dcfbff;font-weight:650}.stButton>button:hover{border-color:#67e8f9;color:white}
textarea,input{background:#091321!important;color:#e8eef8!important;border-color:var(--line)!important}
@media(max-width:800px){.flow{grid-template-columns:1fr}.hero-title{font-size:2.3rem}}
</style>
""",
    unsafe_allow_html=True,
)

SAMPLES = {
    "Product launch": """Northstar Labs launched Atlas, a compact AI assistant for field engineers, in Bengaluru on Tuesday. The company says Atlas can summarize equipment manuals, extract part numbers and answer questions from maintenance notes. Early pilot teams reported faster document lookup, although two teams said the system occasionally missed uncommon component names. Northstar plans a broader rollout in October and will publish an evaluation report before deployment expands.""",
    "Mixed review": """I wanted to love the new service. The onboarding was genuinely smooth and the support team replied within minutes. However, the mobile app crashed twice during checkout and the final invoice was confusing. I would probably use the service again because the staff were excellent, but the product still needs work.""",
    "Research brief": """Researchers evaluated a language model on clinical note summarization across three hospitals. The model reduced average note length substantially while retaining most medication and diagnosis mentions. Performance varied across specialties, and the authors warned that rare details were more likely to be omitted. They recommend human review for high-stakes use and propose entity-retention checks as an additional safety signal.""",
}


def sentiment_color(label: str) -> str:
    return {"Positive": "#5eead4", "Negative": "#fb7185", "Neutral": "#94a3b8"}.get(label, "#94a3b8")


def sentiment_figure(rows: list[dict]):
    x = [f"S{row['id']}" for row in rows]
    y = [row["polarity"] for row in rows]
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line={"width": 2}, marker={"size": 8}))
    fig.add_hline(y=0, line_dash="dot", opacity=.4)
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#91a3bb"},
        xaxis_title="Source sentence",
        yaxis_title="Polarity",
        yaxis_range=[-1, 1],
    )
    return fig


with st.sidebar:
    st.markdown("### ◈ TextScope")
    st.caption("Evidence-aware document intelligence")
    st.markdown("---")
    keyword_count = st.slider("Key concepts", 6, 18, 10)
    summary_sentences = st.slider("Extractive summary sentences", 1, 6, 3)
    summary_mode = st.selectbox("Summary mode", ["auto", "extractive", "abstractive"], index=0)
    qa_threshold = st.slider("QA evidence threshold", 0.05, 0.50, 0.12, 0.01)
    st.markdown("---")
    st.caption("Try a document")
    for label, sample in SAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["text"] = sample
            st.rerun()
    st.markdown("---")
    st.caption("spaCy · TextBlob · BART · RoBERTa · Streamlit")

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Evidence-aware NLP workbench</div>
  <div class="hero-title">TextScope</div>
  <div class="hero-copy">Turn one document into a connected set of NLP views — structure, tone, entities, key concepts, summaries and grounded answers — while keeping the source sentences visible so the output can be inspected rather than blindly trusted.</div>
</div>
<div class="flow">
  <div class="flow-card"><b>1 · Understand</b><span>Measure structure, sentiment trajectory, entities and salient concepts.</span></div>
  <div class="flow-card"><b>2 · Compress</b><span>Create a summary and show keyword/entity coverage diagnostics.</span></div>
  <div class="flow-card"><b>3 · Verify</b><span>Ask questions and return the supporting source sentence — or abstain.</span></div>
</div>
""",
    unsafe_allow_html=True,
)

text = st.text_area(
    "Document",
    value=st.session_state.get("text", ""),
    height=220,
    placeholder="Paste an article, report, review, memo or other English text…",
)

left, right = st.columns([4, 1])
with left:
    analyse = st.button("Analyze document", use_container_width=True)
with right:
    clear = st.button("Clear", use_container_width=True)
if clear:
    st.session_state["text"] = ""
    st.session_state.pop("report", None)
    st.rerun()

if analyse:
    if len(text.split()) < 10:
        st.warning("Provide at least 10 words so the document-level views are meaningful.")
    else:
        with st.spinner("Building document views…"):
            try:
                st.session_state["report"] = core.analyse_document(text, keyword_count=keyword_count)
                st.session_state["analysed_text"] = text
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

report = st.session_state.get("report")
analysed_text = st.session_state.get("analysed_text", "")

if report and analysed_text == text:
    profile = report["profile"]
    sentiment = report["sentiment"]
    entities = report["entities"]
    keywords = report["keywords"]["keywords"]
    sentences = report["sentences"]

    st.markdown("---")
    tabs = st.tabs(["Overview", "Summary", "Entities & concepts", "Ask the document", "Syntax"])

    with tabs[0]:
        st.markdown("### Document fingerprint")
        st.caption("A compact description of the text before any generative model is involved.")
        cols = st.columns(6)
        cols[0].metric("Words", profile["words"])
        cols[1].metric("Sentences", profile["sentences"])
        cols[2].metric("Reading time", f"{profile['reading_minutes']} min")
        cols[3].metric("Avg sentence", f"{profile['avg_sentence_words']} words")
        cols[4].metric("Lexical diversity", profile["lexical_diversity"], help="Unique content lemmas divided by content-token count. Higher usually means more varied vocabulary.")
        cols[5].metric("Entity density", f"{profile['entity_density']}%", help="Named entities per 100 words.")

        c1, c2 = st.columns([1, 2])
        with c1:
            color = sentiment_color(sentiment["label"])
            st.markdown(
                f"<div class='panel'><div class='small'>Overall lexical sentiment</div><div style='font-size:2rem;font-weight:800;color:{color};margin:.25rem 0'>{sentiment['label']}</div><div class='small'>Polarity {sentiment['polarity']:+.3f} · subjectivity {sentiment['subjectivity']:.2f}<br>Intensity {sentiment['intensity']:.1f}/100 · sentence variation {sentiment['sentence_variation']:.3f}</div></div>",
                unsafe_allow_html=True,
            )
            st.caption("Intensity is the magnitude of TextBlob polarity — not a calibrated probability.")
        with c2:
            st.plotly_chart(sentiment_figure(sentiment["sentences"]), use_container_width=True, config={"displayModeBar": False})
            st.caption("The trajectory prevents a mixed document from being reduced to a single positive/negative label.")

        with st.expander("Inspect source sentences", expanded=False):
            for row in sentences:
                sc = sentiment_color(row["label"])
                st.markdown(
                    f"<div class='panel' style='padding:10px 13px;margin:7px 0;border-left:3px solid {sc}'><span class='sent-id'>S{row['id']}</span>{html.escape(row['text'])}</div>",
                    unsafe_allow_html=True,
                )

    with tabs[1]:
        st.markdown("### Compress, then check what survived")
        st.caption("The summary is paired with simple coverage diagnostics so compression is not treated as automatically faithful.")
        if profile["words"] < 30:
            st.info("Use at least 30 words for a meaningful summary.")
        else:
            if st.button("Generate summary", use_container_width=True):
                with st.spinner("Generating and checking summary…"):
                    try:
                        st.session_state["summary_result"] = core.summarize_text(
                            text,
                            num_sentences=summary_sentences,
                            mode=summary_mode,
                        )
                    except Exception as exc:
                        st.error(f"Summary failed: {exc}")
            result = st.session_state.get("summary_result")
            if result:
                st.markdown(f"<div class='insight'>{html.escape(result['summary'])}</div>", unsafe_allow_html=True)
                a, b, c, d = st.columns(4)
                a.metric("Method", result["mode"])
                b.metric("Compression", f"{result['compression_rate']}%")
                c.metric("Entity retention", f"{result['entity_retention']}%", help="Share of unique source named entities that also appear in the summary. This is a diagnostic, not a factuality guarantee.")
                d.metric("Keyword coverage", f"{result['keyword_coverage']}%", help="Share of the top source concepts appearing in the summary.")
                st.caption(result["method"] + ". Coverage measures are warning signals, not proof that every statement is correct.")

    with tabs[2]:
        st.markdown("### Entities and salient concepts")
        st.caption("Every extracted item retains the sentence IDs where it appeared, making the analysis traceable to the document.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Named entities")
            if not entities["entities"]:
                st.info("No named entities detected.")
            else:
                for entity in entities["entities"]:
                    st.markdown(
                        f"<span class='pill'>{html.escape(entity['text'])} · {entity['label']} · S{entity['sentence_id']}</span>",
                        unsafe_allow_html=True,
                    )
        with c2:
            st.markdown("#### Key concepts")
            if not keywords:
                st.info("No strong keyword candidates detected.")
            else:
                table = pd.DataFrame(
                    {
                        "concept": [row["term"] for row in keywords],
                        "score": [row["score"] for row in keywords],
                        "mentions": [row["count"] for row in keywords],
                        "evidence": [", ".join(f"S{x}" for x in row["sentence_ids"]) for row in keywords],
                    }
                )
                st.dataframe(table, use_container_width=True, hide_index=True)
        with st.expander("How concept scoring works"):
            st.write("TextScope forms noun-phrase and noun/proper-noun/adjective candidates, scores them with sentence-aware TF-IDF-style weighting, and gives multi-word concepts a small phrase bonus. The score is document-relative salience, not universal importance.")

    with tabs[3]:
        st.markdown("### Ask the document")
        st.caption("Extractive RoBERTa QA returns a text span plus its source sentence. Low-evidence answers are rejected instead of forced.")
        question = st.text_input("Question", placeholder="What did the authors recommend?", key="qa_question")
        if st.button("Find supported answer", use_container_width=True):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Searching for evidence…"):
                    try:
                        st.session_state["qa_result"] = core.answer_question(text, question, threshold=qa_threshold)
                    except Exception as exc:
                        st.error(f"Question answering failed: {exc}")
        qa = st.session_state.get("qa_result")
        if qa:
            if qa["answer_found"]:
                st.markdown(f"<div class='insight'><b>Answer</b><br>{html.escape(qa['answer'])}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='evidence'><b>Evidence · S{qa['evidence_sentence_id']}</b><br>{html.escape(qa['evidence'])}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Model span score: {qa['confidence']:.1f}% · acceptance threshold: {qa['threshold']:.2f}. This score is model confidence for an extractive span, not a guarantee of correctness.")
            else:
                st.warning(qa["answer"])
                st.caption(f"Best span score {qa['confidence']:.1f}% was below the current evidence requirement or could not be mapped cleanly to a source sentence.")

    with tabs[4]:
        st.markdown("### Sentence structure")
        st.caption("A dependency parse exposes which words modify or depend on others. For readability, TextScope visualizes only the first sentence.")
        try:
            parsed = core.dependency_parse(text)
            st.caption(f"Sentence: {parsed['sentence']}")
            st.components.v1.html(parsed["svg"], height=420, scrolling=True)
            st.dataframe(pd.DataFrame(parsed["tokens"]), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Dependency parse failed: {exc}")
else:
    st.markdown("<div class='panel'><b>Start with a document.</b><div class='small' style='margin-top:6px'>The fast analysis path uses linguistic models only. Transformer models are loaded only when you request abstractive summarization or document Q&A.</div></div>", unsafe_allow_html=True)
