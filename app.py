"""Local web app for the trained models.

    pip install -r requirements.txt
    streamlit run app.py

Opens in the browser at http://localhost:8501. Needs the checkpoints written by the
pipeline (run_all.ps1). Set FINLLM_SMOKE=1 to try it with the tiny smoke-test models.
Page components live in webapp/ui.py, styles in webapp/style.css, and the theme and
fonts in .streamlit/config.toml.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import streamlit as st  # noqa: E402

from finllm import inference  # noqa: E402
from finllm.utils import device_name, get_device  # noqa: E402
from webapp import ui  # noqa: E402

AUTHOR = "Manthan Gehlot"
REPO = "https://github.com/gehlotmanthan/finance-llm-from-scratch"
SMOKE = os.environ.get("FINLLM_SMOKE") == "1"
CK = ROOT / "checkpoints" / ("smoke" if SMOKE else "")

# key: (what it is, file, script that creates it)
MODELS = {
    "sentiment": ("Sentiment classifier, GPT-2 small", "sentiment-best.pth", "scripts/sentiment_classifier.py"),
    "baseline": ("TF-IDF + logistic regression", "tfidf_lr.joblib", "scripts/sentiment_classifier.py"),
    "instruct": ("Finance assistant, GPT-2 medium + LoRA", "instruct.pth", "scripts/instruction_tuning.py"),
    "dapt": ("GPT-2 small, finance pretraining", "gpt2-small-dapt.pth", "scripts/domain_adaptation.py"),
    "scratch": ("Small GPT trained from scratch", "fin-small-scratch.pth", "scripts/pretrain_small_gpt.py"),
}
LM_CHOICES = {
    "GPT-2 small with finance pretraining (124M)": "dapt",
    "Small GPT trained from scratch on headlines (49M)": "scratch",
}
EXAMPLE_HEADLINES = """Tesla shares fall after the company misses quarterly delivery estimates
Apple beats on earnings and raises full-year guidance
Board to hold annual general meeting on 12 May
Bank cuts dividend after loan losses double
Operating profit rose to EUR 13.1 mn from EUR 8.7 mn"""


def path(key: str) -> Path:
    return CK / MODELS[key][1]


st.set_page_config(page_title="Finance LLM from Scratch", page_icon=":material/stacked_line_chart:",
                   layout="wide", initial_sidebar_state="collapsed")
device = get_device()
state = st.session_state


def ready(key: str) -> bool:
    """True if the checkpoint exists; otherwise says how to create it."""
    if path(key).exists():
        return True
    what, fname, script = MODELS[key]
    st.html(ui.note(f"This needs <code>checkpoints/{fname}</code>, which isn't there yet. "
                    f"Create it with <code>python {script}</code>, then reload this page."))
    return False


@st.cache_resource(show_spinner="Loading the sentiment model...")
def sentiment_model():
    return inference.load_sentiment_model(path("sentiment"), device)


@st.cache_resource(show_spinner=False)
def baseline_model():
    return joblib.load(path("baseline")) if path("baseline").exists() else None


@st.cache_resource(show_spinner="Loading the finance assistant (about 1.7 GB, first time only)...")
def instruct_model():
    return inference.load_instruct_model(path("instruct"), device)


@st.cache_resource(show_spinner="Loading the language model...")
def language_model(key):
    return inference.load_language_model(path(key), device)


@st.cache_data(show_spinner=False)
def results():
    return ui.load_results(ROOT / "results")


# ------------------------------------------------------------------ page top
dev = device_name(device)
st.html(ui.css())
st.html(ui.masthead(AUTHOR, "CPU" if dev == "cpu" else dev, REPO))
if SMOKE:
    st.html(ui.smoke_banner())
res = results()
st.html(ui.hero(res))
if band := ui.band(res):
    st.html(band)

st.html(ui.section("Try the models", "Type your own headlines, questions and prompts. "
                   "Each tab runs one of the trained models on this machine."))
tab_sent, tab_ask, tab_gen, tab_att = st.tabs(
    ["Headline sentiment", "Finance assistant", "Text generation", "Attention"])

# ------------------------------------------------------------------ sentiment
with tab_sent:
    st.html(ui.heading("Score headlines",
                       "The fine-tuned GPT-2 and the TF-IDF baseline read the same headlines. "
                       "Each score is P(positive) − P(negative), the number the backtest trades on."))
    text = st.text_area("Headlines, one per line", EXAMPLE_HEADLINES, height=200)
    if st.button("Score headlines", type="primary") and ready("sentiment"):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            state.pop("sent_out", None)
            st.html(ui.note("Type at least one headline, one per line."))
        else:
            clf, ck = sentiment_model()
            rows = inference.classify(clf, ck, lines, device)
            base = baseline_model()
            base_rows = inference.classify_baseline(base, lines, ck["labels"]) if base is not None else None
            state.sent_out = (rows, base_rows, ck["labels"], ck.get("name", "GPT-2 classifier"))
    if "sent_out" in state:
        st.html(ui.sentiment_table(*state.sent_out))

# ------------------------------------------------------------------ assistant
with tab_ask:
    st.html(ui.heading("Ask a question",
                       "GPT-2 medium (355M parameters) tuned with LoRA on finance questions and sentiment "
                       "instructions. It's a small 2019-era model, so expect short answers that are "
                       "sometimes repetitive or wrong."))
    left, right = st.columns([3, 2], gap="large")
    with left:
        instruction = st.text_input("Instruction", "What is a dividend?")
        input_text = st.text_area(
            "Input (optional)", "", height=90,
            placeholder="A headline to classify, for example",
            help="Try the instruction 'Classify the sentiment of this financial news sentence as "
                 "positive, negative or neutral.' with a headline here.")
    with right:
        max_new = st.slider("Max new tokens", 10, 200, 100, key="ask_len")
        temp = st.slider("Temperature (0 picks the likeliest word every time)", 0.0, 1.5, 0.0, 0.1, key="ask_temp")
        topk = st.slider("Top-k (used when temperature is above 0)", 1, 100, 40, key="ask_topk")
    if st.button("Get answer", type="primary") and ready("instruct"):
        model, ck = instruct_model()
        with st.spinner("Writing the answer..."):
            state.ask_out = inference.answer(model, ck["cfg"], instruction, input_text, device, max_new,
                                             temperature=temp, top_k=topk if temp > 0 else None)
    if "ask_out" in state:
        st.html(ui.answer(state.ask_out))

# ------------------------------------------------------------------ generation
with tab_gen:
    st.html(ui.heading("Continue a sentence",
                       "Give a language model the start of a sentence and it writes the rest, one token at a time."))
    choice = st.radio("Model", list(LM_CHOICES), horizontal=True)
    prompt = st.text_input("Start of the sentence", "Shares of Apple rose after")
    c1, c2, c3 = st.columns(3, gap="large")
    n = c1.slider("New tokens", 5, 100, 30)
    t = c2.slider("Temperature", 0.0, 1.5, 0.8, 0.1)
    k = c3.slider("Top-k", 1, 100, 40)
    if st.button("Continue the sentence", type="primary") and ready(LM_CHOICES[choice]):
        model, cfg = language_model(LM_CHOICES[choice])
        out = inference.complete(model, cfg, prompt, device, n, temperature=t, top_k=k if t > 0 else None)
        state.gen_out = (prompt, out)
    if "gen_out" in state:
        st.html(ui.continuation(*state.gen_out))

# ------------------------------------------------------------------ attention
with tab_att:
    st.html(ui.heading("See what the model attends to",
                       "Each row is a token looking back over the tokens before it, and darker squares "
                       "get more of its attention. Many heads park most of their weight on the first "
                       "token, known as an attention sink, so the shading is scaled to the other columns."))
    att_text = st.text_input("Headline", "Tesla shares fall after the company misses quarterly delivery estimates")
    if ready("dapt") and att_text.strip():
        model, cfg = language_model("dapt")
        tokens, maps = inference.attention_maps(model, att_text, device)
        c1, c2 = st.columns(2, gap="large")
        layer = c1.slider("Layer", 0, len(maps) - 1, min(2, len(maps) - 1))
        head_opts = ["Average of all heads"] + [f"Head {h}" for h in range(maps[layer].shape[0])]
        head = c2.selectbox("Head", head_opts, index=min(9, len(head_opts) - 1))
        mat = maps[layer].mean(0) if head == head_opts[0] else maps[layer][head_opts.index(head) - 1]
        st.html(ui.attention_grid(tokens, mat, layer, head.lower()))

# ------------------------------------------------------------------ colophon
st.html(ui.colophon([(what, fname, path(k).exists(), script) for k, (what, fname, script) in MODELS.items()], REPO))
