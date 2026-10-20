"""HTML pieces for app.py.

Everything the page shows that isn't a Streamlit widget is built here, so app.py
reads as layout. Styles are in style.css next to this file. All user text is
escaped before it goes into the page.
"""

from __future__ import annotations

import json
from datetime import date
from html import escape
from pathlib import Path

import numpy as np

STYLE = Path(__file__).with_name("style.css")


def css() -> str:
    return f"<style>{STYLE.read_text(encoding='utf-8')}</style>"


def wrap(inner: str, tag: str = "div", cls: str = "") -> str:
    classes = f"fl {cls}".strip()
    return f'<{tag} class="{classes}">{inner}</{tag}>'


def signed(x: float, digits: int = 2) -> str:
    """+0.36 / −0.42 with a real minus sign."""
    return f"{x:+.{digits}f}".replace("-", "−")


def month_year(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%b %Y")


# ------------------------------------------------------------------ results
def load_results(results_dir: Path) -> dict:
    """The headline numbers for the page, read from the experiment JSON files.

    Missing files just leave their keys out, so a partial run still renders.
    """
    def read(name):
        p = Path(results_dir) / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    out: dict = {}
    sent = read("sentiment_classifier")
    if sent and not sent.get("smoke"):
        sel = sent["selected_by_validation"]
        out["f1"] = sent["models"][sel]["test"]["macro_f1"]
        out["f1_baseline"] = sent["models"]["TF-IDF + LogReg"]["test"]["macro_f1"]
        out["n_test"] = sent["data"]["test"]["n"]
    ins = read("instruction_tuning")
    if ins and not ins.get("smoke"):
        out["instruct_acc"] = ins["sentiment_after"]["accuracy"]
        out["instruct_acc_before"] = ins["sentiment_before"]["accuracy"]
    bt = read("backtest")
    if bt and not bt.get("smoke"):
        o = bt["results"]["GPT-2 classifier"]["out_of_sample"]
        same = o["ic"]["same_day_close_to_close (not tradable)"]
        out["ic_same_day"], out["ic_same_day_t"] = same["mean_ic"], same["nw_tstat"]
        t = o["avg_daily_turnover"]
        out["breakeven_bps"] = o["gross"]["mean_daily_bps"] / t if t else None
        out["periods"] = bt["periods"]
    sc = read("headline_scoring")
    if sc and not sc.get("smoke"):
        out["n_headlines"] = sc["n_headlines"]
    return out


# ------------------------------------------------------------------ page top
def masthead(author: str, device: str, repo_url: str) -> str:
    return wrap(
        f'<span>A research project by {escape(author)}</span>'
        f'<span class="fl-masthead-side">'
        f'<span class="fl-quiet">Running on {escape(device)}</span>'
        f'<a href="{escape(repo_url)}" target="_blank" rel="noopener">Code on GitHub</a>'
        f'</span>', "header", "fl-masthead")


def smoke_banner() -> str:
    return wrap("Smoke mode: these are tiny untrained test models, so their outputs are noise.",
                "div", "fl-smoke")


def hero(res: dict) -> str:
    n = res.get("n_headlines")
    tested = f"tested as a trading signal on {n:,} headlines" if n else "tested as a trading signal"
    lead = (f"A GPT-2-style language model written from scratch in PyTorch, taught to read "
            f"financial news on a 6 GB laptop GPU, then {tested}: out of sample, against a simple "
            f"baseline, and after costs.")
    return wrap(
        '<h1 class="fl-title"><span>Finance LLM</span><span>from Scratch</span></h1>'
        f'<p class="fl-lead">{escape(lead)}</p>'
        + (calendar(res["periods"]) if "periods" in res else ""),
        "section", "fl-hero")


def calendar(periods: dict) -> str:
    """The news calendar as a thick rule, split where the data changes role."""
    cuts = [periods["start"], periods["pretrain_end"], periods["is_end"], periods["end"]]
    days = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(cuts, cuts[1:])]
    segs = [("Pretraining text", "Read by the models to learn finance language. Never traded."),
            ("In-sample", "First look at the backtest, where any design choice would be made."),
            ("Out-of-sample", "Run once, with every setting already fixed in code.")]
    cols = " ".join(f"{d}fr" for d in days)
    body = "".join(
        f'<div class="fl-cal-seg"><span class="fl-cal-name">{name}</span>'
        f'<span class="fl-cal-dates">{month_year(a)} to {month_year(b)}</span>'
        f'<span class="fl-cal-use">{use}</span></div>'
        for (name, use), a, b in zip(segs, cuts, cuts[1:]))
    return (f'<figure class="fl-cal" aria-label="How the news calendar is split">'
            f'<figcaption class="fl-cal-caption">How the ten years of headlines are used</figcaption>'
            f'<div class="fl-cal-track" style="grid-template-columns:{cols}">{body}</div></figure>')


def band(res: dict) -> str:
    stats = []
    if "f1" in res:
        stats.append((f"{res['f1']:.3f}",
                      f"Sentiment macro-F1 on {res['n_test']:,} held-out sentences. "
                      f"The TF-IDF baseline scores {res['f1_baseline']:.3f}."))
    if "instruct_acc" in res:
        stats.append((f"{res['instruct_acc']:.0%}",
                      f"Sentiment questions answered correctly by GPT-2 medium after LoRA tuning, "
                      f"up from {res['instruct_acc_before']:.1%}."))
    if "ic_same_day" in res:
        stats.append((f"{res['ic_same_day']:.3f}",
                      f"Same-day rank correlation between headline sentiment and returns, "
                      f"out of sample (t = {res['ic_same_day_t']:.1f})."))
    if res.get("breakeven_bps") is not None:
        stats.append((f"{res['breakeven_bps']:.1f} bp",
                      "Cost per side at which trading at the next open breaks even. "
                      "Real costs are higher."))
    if not stats:
        return ""
    cells = "".join(f'<div class="fl-stat"><p class="fl-stat-num">{escape(num)}</p>'
                    f'<p class="fl-stat-text">{escape(text)}</p></div>' for num, text in stats)
    finding = ""
    if "ic_same_day" in res:
        finding = ('<p class="fl-band-finding">The model reads the news the way the market does. '
                   'By the next morning’s open, the market has already priced it in.</p>')
    return wrap(f'<h2 class="fl-band-title">What the full run found</h2>'
                f'<div class="fl-band-grid">{cells}</div>{finding}', "section", "fl-band")


def section(title: str, intro: str = "") -> str:
    return wrap(f'<h2 class="fl-h2">{escape(title)}</h2>'
                + (f'<p class="fl-intro">{escape(intro)}</p>' if intro else ""), "div", "fl-section")


def heading(title: str, desc: str = "") -> str:
    return wrap(f'<h3 class="fl-h3">{escape(title)}</h3>'
                + (f'<p class="fl-desc">{escape(desc)}</p>' if desc else ""))


def note(html_text: str) -> str:
    """A bordered notice. Takes trusted HTML (built by the app, not user input)."""
    return wrap(html_text, "div", "fl-note")


def caption(text: str) -> str:
    return wrap(f'<p class="fl-caption">{escape(text)}</p>')


# ------------------------------------------------------------------ sentiment
def _bar(score: float) -> str:
    w = min(abs(score), 1.0) * 50
    side = "left" if score >= 0 else "right"
    return f'<span class="fl-bar" aria-hidden="true"><span style="{side}:50%;width:{w:.1f}%"></span></span>'


def _verdict(label: str, score: float, title: str = "") -> str:
    t = f' title="{escape(title)}"' if title else ""
    return (f'<div class="fl-verdict"{t}><span class="fl-verdict-label">{escape(label)}</span>'
            f'{_bar(score)}<span class="fl-verdict-score">{signed(score)}</span></div>')


def sentiment_table(rows: list[dict], baseline: list[dict] | None, labels: list[str], model_name: str) -> str:
    has_base = bool(baseline)
    head = "<th scope='col'>Headline</th><th scope='col'>GPT-2</th>" + (
        "<th scope='col'>TF-IDF baseline</th>" if has_base else "")
    body, n_disagree = [], 0
    for i, r in enumerate(rows):
        probs = ", ".join(f"{l} {r[l]:.2f}" for l in labels)
        flag = ""
        if has_base and baseline[i]["label"] != r["label"]:
            n_disagree += 1
            flag = '<span class="fl-flag" aria-hidden="true"></span><span class="fl-sr"> (the models disagree)</span>'
        cells = f"<td data-label='GPT-2'>{_verdict(r['label'], r['score'], 'GPT-2 probabilities: ' + probs)}</td>"
        if has_base:
            cells += f"<td data-label='TF-IDF'>{_verdict(baseline[i]['label'], baseline[i]['score'])}</td>"
        body.append(f"<tr><th scope='row'>{escape(r['text'])}{flag}</th>{cells}</tr>")
    notes = [f"Score = P(positive) − P(negative), from −1 to +1. "
             f"GPT-2 model: {escape(model_name)}. Hover a GPT-2 score for its three probabilities."]
    if n_disagree:
        notes.insert(0, '<span class="fl-flag" aria-hidden="true"></span>&nbsp; The two models give different labels.')
    note_html = "".join(f'<p class="fl-caption">{n}</p>' for n in notes)
    return wrap(f'<div class="fl-table-wrap"><table class="fl-table is-stack"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>{note_html}')


# ------------------------------------------------------------------ generated text
def answer(text: str) -> str:
    text = text.strip()
    if not text:
        return note("The model returned an empty answer. Raise the temperature or allow more tokens and try again.")
    first, rest = text[0], text[1:]
    if first.isalnum():
        body = f'<span class="fl-dropcap" aria-hidden="true">{escape(first)}</span>' \
               f'<span class="fl-sr">{escape(first)}</span>{escape(rest)}'
    else:
        body = escape(text)
    return wrap(f'<p class="fl-answer">{body}</p>')


def continuation(prompt: str, text: str) -> str:
    return wrap(f'<p class="fl-continuation">{escape(prompt)}'
                f'<span class="fl-square fl-seam" role="img" aria-label="the model continues from here"></span>'
                f'{escape(text.lstrip())}</p>'
                f'<p class="fl-caption">Your prompt ends at the square. Everything after it is the model’s.</p>')


# ------------------------------------------------------------------ attention
def attention_grid(tokens: list[str], mat: np.ndarray, layer: int, head: str) -> str:
    """Rows are tokens reading back over earlier tokens; darker means more attention.

    The shade scale is set by everything except the first column, because many heads
    park most of their weight on the first token (an attention sink) and it would
    wash out the rest. The first column is clipped to full black.
    """
    labels = [t.strip() or "␣" for t in tokens]
    n = len(labels)
    vmax = float(mat[:, 1:].max()) if n > 1 else 1.0
    vmax = max(vmax, 1e-6)
    parts = ['<span></span>'] + [f'<span class="fl-att-col">{escape(l)}</span>' for l in labels]
    for i in range(n):
        parts.append(f'<span class="fl-att-row">{escape(labels[i])}</span>')
        for j in range(n):
            if j > i:
                parts.append('<span class="fl-att-cell is-masked"></span>')
                continue
            w = float(mat[i, j])
            a = min(w / vmax, 1.0)
            light = " is-light" if a < 0.08 else ""
            tip = f"{labels[i]} → {labels[j]}: {w:.3f}"
            parts.append(f'<span class="fl-att-cell{light}" style="--a:{a:.3f}" title="{escape(tip)}"></span>')
    cell = "2.5rem" if n <= 14 else "2rem" if n <= 22 else "1.6rem"
    desc = f"Attention weights for layer {layer}, {head}, over {n} tokens"
    scale = "".join(f'<span class="fl-att-cell{" is-light" if a < 0.08 else ""}" style="--a:{a}"></span>'
                    for a in (0, 0.25, 0.5, 0.75, 1))
    legend = (f'<div class="fl-att-legend">'
              f'<span class="fl-att-scale"><span class="fl-att-scale-label">Less</span>{scale}'
              f'<span class="fl-att-scale-label">More attention</span></span>'
              f'<span class="fl-att-scale"><span class="fl-att-cell is-masked" style="--cell:1rem"></span>'
              f'<span class="fl-att-scale-label">Future token, hidden by the causal mask</span></span></div>')
    return wrap(f'<div class="fl-att-wrap"><div class="fl-att" role="img" aria-label="{escape(desc)}" '
                f'style="--n:{n};--cell-size:{cell}">{"".join(parts)}</div></div>{legend}')


# ------------------------------------------------------------------ colophon
def colophon(models: list[tuple[str, str, bool, str]], repo_url: str) -> str:
    """models: (what it is, file name, file exists, script that creates it)."""
    rows = "".join(
        f"<tr><th scope='row'>{escape(what)}</th><td><code>{escape(fname)}</code></td>"
        f"<td>{'Ready' if ok else f'Not found. Run <code>python {escape(script)}</code>'}</td></tr>"
        for what, fname, ok, script in models)
    return wrap(
        '<h2 class="fl-h2">About this app</h2>'
        '<div class="fl-colophon-grid">'
        f'<div><table class="fl-table"><thead><tr><th scope="col">Model</th><th scope="col">File</th>'
        f'<th scope="col">Status</th></tr></thead><tbody>{rows}</tbody></table></div>'
        '<div>'
        '<p>The models load from <code>checkpoints/</code> and run on this machine. '
        'Nothing you type leaves it.</p>'
        '<p>The model code follows Sebastian Raschka’s <em>Build a Large Language Model '
        '(From Scratch)</em>. The finance data pipeline, experiments, statistics and backtest '
        'are this project’s own.</p>'
        f'<p><a href="{escape(repo_url)}" target="_blank" rel="noopener">Code, tests and full results on GitHub</a></p>'
        '</div></div>', "footer", "fl-section fl-colophon")
