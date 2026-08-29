"""Hosted TextScope entrypoint with an editorial research-lab visual identity."""

from pathlib import Path
import runpy

import streamlit as st

APP = Path(__file__).resolve().parent / "textscope_app.py"
runpy.run_path(str(APP), run_name="__main__")

# Hosted visual layer: intentionally light/editorial so TextScope does not look
# like the dark monitoring dashboards elsewhere in the portfolio.
st.markdown(
    """
<style>
:root{--paper:#f5f1e8;--paper-2:#ebe4d7;--card:#fffdf8;--ink:#25231f;--muted:#6f6a61;--line:#d9d0c2;--blue:#315f7d;--violet:#705b76;}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#faf7f0 0%,var(--paper) 46%,#f1eadf 100%)!important;color:var(--ink)!important;}
[data-testid="stSidebar"]{background:#e9e0d2!important;border-right:1px solid #cfc3b3!important;}
[data-testid="stSidebar"] *{color:#39352f!important;}
.block-container{max-width:1260px!important;padding-top:2.4rem!important;}
h1,h2,h3,h4,p,li,label{color:var(--ink)!important;}
.hero{padding:1rem 0 .8rem!important;border-bottom:1px solid var(--line);margin-bottom:.9rem!important;}
.eyebrow{color:var(--blue)!important;letter-spacing:.16em!important;}
.hero-title{background:linear-gradient(90deg,#1f2b33,#315f7d 55%,#705b76)!important;-webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;font-family:Georgia,'Times New Roman',serif!important;font-weight:700!important;letter-spacing:-.04em!important;}
.hero-copy{color:#625d55!important;max-width:900px!important;font-family:Georgia,'Times New Roman',serif!important;font-size:1.04rem!important;}
.flow{gap:14px!important;margin-top:20px!important;}
.flow-card,.panel{background:rgba(255,253,248,.94)!important;border:1px solid var(--line)!important;border-radius:8px!important;box-shadow:0 5px 18px rgba(76,65,50,.055)!important;}
.flow-card{border-top:3px solid #315f7d!important;}
.flow-card:nth-child(2){border-top-color:#9a784c!important;}.flow-card:nth-child(3){border-top-color:#705b76!important;}
.flow-card b{color:#302d28!important;font-family:Georgia,'Times New Roman',serif!important;}.flow-card span,.small{color:#777066!important;}
.insight{border-left:4px solid #315f7d!important;background:#edf2f3!important;color:#2f383c!important;border-radius:2px 8px 8px 2px!important;}
.evidence{border-left:4px solid #705b76!important;background:#f1edf2!important;color:#413745!important;border-radius:2px 8px 8px 2px!important;}
.pill{border:1px solid #c7b9a7!important;background:#f2eadf!important;color:#4d443a!important;border-radius:5px!important;}
.sent-id{color:#315f7d!important;}
[data-testid="stMetric"]{background:#fffdf8!important;border:1px solid var(--line)!important;border-radius:7px!important;box-shadow:none!important;}
[data-testid="stMetricLabel"]{color:#7a7369!important;}
[data-testid="stMetricValue"]{color:#292621!important;font-family:Georgia,'Times New Roman',serif!important;}
.stTabs [data-baseweb="tab-list"]{border-bottom:1px solid var(--line)!important;gap:2px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#7a7369!important;border-radius:0!important;border-bottom:2px solid transparent!important;}
.stTabs [aria-selected="true"]{background:transparent!important;color:#315f7d!important;border-bottom-color:#315f7d!important;}
.stButton>button{background:#315f7d!important;color:#fffdf8!important;border:1px solid #315f7d!important;border-radius:6px!important;box-shadow:none!important;}
.stButton>button:hover{background:#274e68!important;border-color:#274e68!important;color:white!important;transform:none!important;}
textarea,input{background:#fffdf8!important;color:#2c2924!important;border:1px solid #cfc4b5!important;border-radius:6px!important;}
[data-baseweb="select"]>div{background:#fffdf8!important;border-color:#cfc4b5!important;color:#2c2924!important;}
hr{border-color:#d8cec0!important;}
[data-testid="stExpander"]{background:#fffdf8!important;border:1px solid var(--line)!important;border-radius:7px!important;}
[data-testid="stDataFrame"]{border:1px solid var(--line)!important;border-radius:6px!important;overflow:hidden!important;}
.stCaptionContainer, [data-testid="stCaptionContainer"]{color:#756f66!important;}
@media(max-width:800px){.hero-title{font-size:2.35rem!important}.flow{grid-template-columns:1fr!important}}
</style>
""",
    unsafe_allow_html=True,
)
