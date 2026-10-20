"""Local web app for the trained models.

    pip install -r requirements.txt
    streamlit run app.py

Opens in the browser at http://localhost:8501. Needs the checkpoints written by the
pipeline (run_all.ps1). Set FINLLM_SMOKE=1 to try it with the tiny smoke-test models.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from finllm import inference  # noqa: E402
from finllm.utils import device_name, get_device  # noqa: E402

SMOKE = os.environ.get("FINLLM_SMOKE") == "1"
CK = ROOT / "checkpoints" / ("smoke" if SMOKE else "")
FILES = {
    "sentiment": CK / "sentiment-best.pth",
    "baseline": CK / "tfidf_lr.joblib",
    "instruct": CK / "instruct.pth",
    "scratch": CK / "fin-small-scratch.pth",
    "dapt": CK / "gpt2-small-dapt.pth",
}
MAKES = {
    "sentiment": "scripts/sentiment_classifier.py", "baseline": "scripts/sentiment_classifier.py",
    "instruct": "scripts/instruction_tuning.py", "scratch": "scripts/pretrain_small_gpt.py",
    "dapt": "scripts/domain_adaptation.py",
}
LM_CHOICES = {
    "GPT-2 + finance domain pretraining (124M)": "dapt",
    "Small GPT trained from scratch on headlines (49M)": "scratch",
}
EXAMPLE_HEADLINES = """Tesla shares fall after the company misses quarterly delivery estimates
Apple beats on earnings and raises full-year guidance
Board to hold annual general meeting on 12 May
Bank cuts dividend after loan losses double
Operating profit rose to EUR 13.1 mn from EUR 8.7 mn"""

st.set_page_config(page_title="Finance LLM from Scratch", page_icon="📈", layout="wide")
device = get_device()


def missing(key):
    if not FILES[key].exists():
        st.warning(f"`{FILES[key].relative_to(ROOT)}` not found. Create it with `python {MAKES[key]}`.")
        return True
    return False


@st.cache_resource(show_spinner="Loading the sentiment model...")
def sentiment_model():
    return inference.load_sentiment_model(FILES["sentiment"], device)


@st.cache_resource(show_spinner=False)
def baseline_model():
    return joblib.load(FILES["baseline"]) if FILES["baseline"].exists() else None


@st.cache_resource(show_spinner="Loading the instruction-tuned model (about 1.7 GB)...")
def instruct_model():
    return inference.load_instruct_model(FILES["instruct"], device)


@st.cache_resource(show_spinner="Loading the language model...")
def language_model(key):
    return inference.load_language_model(FILES[key], device)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Finance LLM from Scratch")
    st.caption("A GPT-2-style model built in PyTorch and applied to financial text.")
    st.write(f"Device: **{device_name(device)}**")
    if SMOKE:
        st.info("Smoke mode: tiny untrained test models, so outputs are noise.")
    st.write("Checkpoints:")
    for k, f in FILES.items():
        st.write(("✅ " if f.exists() else "❌ ") + f"`{f.name}`")
    st.caption("[Code and results on GitHub](https://github.com/gehlotmanthan/finance-llm-from-scratch)")

tab_sent, tab_ask, tab_gen, tab_att = st.tabs(
    ["Headline sentiment", "Finance assistant", "Text generation", "Attention viewer"])

# ------------------------------------------------------------------ sentiment
with tab_sent:
    st.subheader("Score financial headlines")
    st.write("Fine-tuned GPT-2 against the TF-IDF + logistic regression baseline. "
             "The score is P(positive) − P(negative), the number the backtest trades on.")
    text = st.text_area("One headline per line", EXAMPLE_HEADLINES, height=150)
    if st.button("Score headlines", type="primary") and not missing("sentiment"):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clf, ck = sentiment_model()
        rows = inference.classify(clf, ck, lines, device)
        df = pd.DataFrame(rows).rename(columns={"text": "headline", "label": "GPT-2 label", "score": "GPT-2 score"})
        base = baseline_model()
        if base is not None:
            b = inference.classify_baseline(base, lines, ck["labels"])
            df["TF-IDF label"] = [x["label"] for x in b]
            df["TF-IDF score"] = [x["score"] for x in b]
        # labels and scores first, the GPT-2 class probabilities after them
        front = ["headline", "GPT-2 label", "GPT-2 score", "TF-IDF label", "TF-IDF score"]
        df = df[[c for c in front if c in df] + [c for c in df if c not in front]]
        st.dataframe(df.style.format({c: "{:.2f}" for c in df.columns if df[c].dtype.kind == "f"}),
                     width="stretch", hide_index=True)
        st.caption(f"Model: {ck.get('name', 'GPT-2 classifier')}")

# ------------------------------------------------------------------ assistant
with tab_ask:
    st.subheader("Ask the instruction-tuned model")
    st.write("GPT-2 medium (355M) tuned with LoRA on finance Q&A and sentiment instructions. "
             "It's a small 2019-era model: expect short, sometimes repetitive or wrong answers.")
    col1, col2 = st.columns([2, 1])
    with col1:
        instruction = st.text_input("Instruction", "What is a dividend?")
        input_text = st.text_area("Input (optional)", "", height=80,
                                  help="E.g. a headline, with the instruction 'Classify the sentiment of this "
                                       "financial news sentence as positive, negative or neutral.'")
    with col2:
        max_new = st.slider("Max new tokens", 10, 200, 100, key="ask_len")
        temp = st.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.0, 0.1, key="ask_temp")
        topk = st.slider("Top-k", 1, 100, 40, key="ask_topk")
    if st.button("Answer", type="primary") and not missing("instruct"):
        model, ck = instruct_model()
        with st.spinner("Generating..."):
            out = inference.answer(model, ck["cfg"], instruction, input_text, device, max_new,
                                   temperature=temp, top_k=topk if temp > 0 else None)
        st.markdown(f"> {out or '_(empty answer)_'}")

# ------------------------------------------------------------------ generation
with tab_gen:
    st.subheader("Continue a prompt")
    choice = st.radio("Model", list(LM_CHOICES), horizontal=True)
    prompt = st.text_input("Prompt", "Shares of Apple rose after")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("New tokens", 5, 100, 30)
    t = c2.slider("Temperature", 0.0, 1.5, 0.8, 0.1)
    k = c3.slider("Top-k", 1, 100, 40)
    if st.button("Generate", type="primary") and not missing(LM_CHOICES[choice]):
        model, cfg = language_model(LM_CHOICES[choice])
        out = inference.complete(model, cfg, prompt, device, n, temperature=t, top_k=k if t > 0 else None)
        st.markdown(f"**{prompt}**{out}")

# ------------------------------------------------------------------ attention
with tab_att:
    st.subheader("What does the model attend to?")
    st.write("Each row is a token, and the colour shows how much it attends to every earlier token. "
             "Many heads park most of their attention on the first token, which is known as an attention sink.")
    att_text = st.text_input("Headline", "Tesla shares fall after the company misses quarterly delivery estimates")
    if not missing("dapt"):
        model, cfg = language_model("dapt")
        tokens, maps = inference.attention_maps(model, att_text, device)
        c1, c2 = st.columns(2)
        layer = c1.slider("Layer", 0, len(maps) - 1, min(2, len(maps) - 1))
        head_opts = ["average of all heads"] + [str(h) for h in range(maps[layer].shape[0])]
        head = c2.selectbox("Head", head_opts, index=min(9, len(head_opts) - 1))
        mat = maps[layer].mean(0) if head == head_opts[0] else maps[layer][int(head)]
        labels = [x.strip() or "␣" for x in tokens]
        fig, ax = plt.subplots(figsize=(0.45 * len(labels) + 2, 0.45 * len(labels) + 1.5))
        cmap = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
        vmax = max(mat[:, 1:].max(), 1e-6) if mat.shape[1] > 1 else 1.0
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(f"Layer {layer}, head {head} (colour scale ignores the first column)", fontsize=10, loc="left")
        st.pyplot(fig, width="content")
        plt.close(fig)
