"""Build reports/RESULTS.md and the README "Results" block from results/*.json.

Every number in the README comes from a results file written by a script, so
the write-up can never drift from what was actually run. Re-run after any
experiment:  python scripts/make_report.py
"""

import argparse
import re
from pathlib import Path

from _common import ROOT
from finllm.utils import load_json

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def f(x, nd=3, pct=False, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    if isinstance(x, str):
        return x
    if pct:
        return f"{x:+.1%}" if sign else f"{x:.1%}"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(pair, nd=3):
    return f"[{pair[0]:.{nd}f}, {pair[1]:.{nd}f}]" if pair and pair[0] == pair[0] else "–"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def img(path, alt):
    return f"![{alt}](../{path})"


def load(res_dir, name):
    p = res_dir / name
    return load_json(p) if p.exists() else None


def sec_ch2(r):
    if not r:
        return "_Not run yet: `python scripts/ch02_tokenization.py`_"
    rows = [[k, f(v["tokens_per_word"], 2), f(v["chars_per_token"], 2), f(v["share_words_split"], pct=True)]
            for k, v in r["bpe_stats"].items()]
    o = r["word_level_oov"]
    terms = ", ".join(f"`{t}` → {len(p)}" for t, p in list(r["finance_terms"].items())[:10])
    return "\n\n".join([
        table(["Corpus", "Tokens / word", "Chars / token", "Words split into 2+ tokens"], rows),
        f"A word-level vocabulary built on general English maps **{f(o['vocab_from_verdict_on_phrasebank']['oov_rate'], pct=True)}** "
        f"of PhraseBank tokens to `<|unk|>`; one built on PhraseBank still loses "
        f"**{f(o['vocab_from_phrasebank_on_headlines']['oov_rate'], pct=True)}** of headline tokens. BPE never fails, "
        f"but it fragments finance vocabulary: {terms}.",
        img(r["figures"][1], "BPE fragmentation of finance terms")])


def sec_ch3(r):
    if not r:
        return "_Not run yet: `python scripts/ch03_attention.py`_"
    rows = [[x["T"], f(x["explicit_ms"], 2), f(x.get("fused_sdpa_ms"), 2),
             f(x["explicit_peak_mb"], 0) if isinstance(x["explicit_peak_mb"], float) else x["explicit_peak_mb"] or "–",
             f(x["fused_sdpa_peak_mb"], 0) if isinstance(x["fused_sdpa_peak_mb"], float) else x["fused_sdpa_peak_mb"] or "–"]
            for x in r["scaling"]]
    return "\n\n".join([
        f"Device: {r['device']}. Explicit attention equals PyTorch's fused kernel to "
        f"{r['max_abs_diff_explicit_vs_fused']:.1e}; editing future tokens changes past outputs by "
        f"{r['causal_leak_max_abs_change']:.1e} (no look-ahead).",
        table(["T", "explicit ms", "fused ms", "explicit peak MB", "fused peak MB"], rows),
        img(r["figures"][0], "attention scaling")])


def sec_ch4(r):
    if not r:
        return "_Not run yet: `python scripts/ch04_model.py`_"
    rows = [[s["model"], f"{s['params_tied']:,}", f(s["fp32_weights_gb"], 2), f(s["adamw_train_state_gb"], 1)]
            for s in r["sizes"]]
    parts = [table(["Model", "Parameters", "fp32 weights (GB)", "AdamW training state (GB)"], rows)]
    if r["train_step_probes"]:
        best = {}
        for p in r["train_step_probes"]:
            if p["status"] == "ok":
                key = (p["model"], "fused" if p["sdpa"] else "explicit", "bf16" if p["bf16"] else "fp32")
                if p["batch"] > best.get(key, (0, 0))[0]:
                    best[key] = (p["batch"], p["tokens_per_s"])
        rows = [[k[0], k[1], k[2], v[0], f"{v[1]:,.0f}"] for k, v in sorted(best.items())]
        parts += [f"Measured on **{r['device']}** (sequence length 256, one AdamW step):",
                  table(["Model", "Attention", "Precision", "Largest batch that fits", "Tokens/s"], rows),
                  img(r["figures"][0], "GPU memory fit")]
    return "\n\n".join(parts)


def sec_ch5(pre, dom, dapt):
    parts = []
    if pre:
        parts.append(
            f"**From scratch.** A {pre['params'] / 1e6:.0f}M-parameter GPT trained on {pre['data']['train_tokens']:,} "
            f"tokens of headlines (≤ {pre['data']['window_end']}) in {pre['train_seconds'] / 60:.1f} min reaches "
            f"validation perplexity **{f(pre['final_val_ppl'], 1)}**, against **{f(pre['unigram_val_ppl'], 1)}** for a "
            f"word-frequency baseline. Sample: _“{pre['samples'][0]['text']}”_\n\n" + img(pre["figures"][0], "pretraining loss"))
    else:
        parts.append("_From-scratch run not done yet: `python scripts/ch05_pretrain_from_scratch.py`_")
    if dom:
        corp = list(next(iter(dom["perplexity"].values())))
        rows = [[m] + [f(d[c]["perplexity"], 1) for c in corp] for m, d in dom["perplexity"].items()]
        sink = dom["attention"]["sink_share_by_layer"]
        parts.append("**OpenAI GPT-2 loaded into this code: perplexity by domain** (256-token windows)\n\n"
                     + table(["Model"] + corp, rows)
                     + f"\n\nGreedy sample: _“{dom['samples'][next(iter(dom['samples']))][0]['greedy']}”_\n\n"
                     f"Attention sink: averaged over heads, layers 2–{len(sink)} put "
                     f"{min(sink[1:]):.0%}–{max(sink[1:]):.0%} of attention on the first token.\n\n"
                     + img(dom["figures"][0], "domain perplexity") + "\n\n" + img(dom["figures"][1], "attention"))
    else:
        parts.append("_Domain-gap run not done yet: `python scripts/ch05_gpt2_domain.py`_")
    if dapt:
        rows = [[c, f(b["perplexity"], 1), f(dapt["perplexity_after"][c]["perplexity"], 1),
                 f(dapt["perplexity_after"][c]["perplexity"] / b["perplexity"] - 1, pct=True, sign=True)]
                for c, b in dapt["perplexity_before"].items()]
        parts.append(f"**Domain-adaptive pretraining** ({dapt['tokens_trained'] / 1e6:.1f}M tokens, "
                     f"{dapt['train_seconds'] / 60:.0f} min)\n\n"
                     + table(["Corpus", "GPT-2", "GPT-2 + DAPT", "Change"], rows) + "\n\n" + img(dapt["figures"][0], "DAPT"))
    else:
        parts.append("_DAPT not run yet: `python scripts/ch05_dapt.py`_")
    return "\n\n".join(parts)


def sec_ch6(r):
    if not r:
        return "_Not run yet: `python scripts/ch06_sentiment.py`_"
    rows = []
    for name, m in r["models"].items():
        t = m["test"]
        d = m.get("vs_tfidf_macro_f1")
        mc = m.get("vs_tfidf_mcnemar")
        rows.append([("**" + name + "**") if name == r["selected_by_validation"] else name,
                     f(t["accuracy"]), f(t["macro_f1"]), ci(t["macro_f1_ci"]),
                     f"{d['diff']:+.3f} {ci([d['ci_low'], d['ci_high']])}" if d else "–",
                     f(mc["p_value"], 4) if mc else "–"])
    sel = r["models"][r["selected_by_validation"]]
    agree = sel.get("by_agreement", {})
    agree_txt = ", ".join(f"{k}% agreement: {f(v['accuracy'])} (n={v['n']})" for k, v in sorted(agree.items(), key=lambda kv: int(kv[0])))
    return "\n\n".join([
        f"Test set: {r['data']['test']['n']:,} PhraseBank sentences (never used for training or model selection). "
        f"Bold = model chosen on validation macro-F1.",
        table(["Model", "Accuracy", "Macro-F1", "95% CI", "Δ macro-F1 vs TF-IDF [95% CI]", "McNemar p"], rows),
        f"Selected model by annotator agreement — {agree_txt}.",
        img(r["figures"][0], "model comparison") + "\n\n" + img(r["figures"][1], "confusion matrix")])


def sec_ch7(r, ch6):
    if not r:
        return "_Not run yet: `python scripts/ch07_instruct.py`_"
    b, a = r["sentiment_before"], r["sentiment_after"]
    rows = [["Base model, zero-shot", f(b["accuracy"]), f(b["macro_f1"]), f(b["invalid_rate"], pct=True)],
            ["After instruction tuning", f(a["accuracy"]), f(a["macro_f1"]), f(a["invalid_rate"], pct=True)]]
    if ch6:
        s = ch6["models"][ch6["selected_by_validation"]]["test"]
        rows.append([f"Ch 6 classifier ({ch6['selected_by_validation']})", f(s["accuracy"]), f(s["macro_f1"]), "0.0%"])
    lora = r.get("lora")
    lora_txt = (f"LoRA rank {lora['rank']}: {lora['trainable_params']:,} trainable parameters "
                f"({lora['trainable_share']:.2%} of {lora['total_params']:,}). ") if lora else "Full fine-tuning. "
    peak = f"Peak GPU memory {r['peak_gpu_gb']:.1f} GB. " if r.get("peak_gpu_gb") else ""
    ex = a["examples"][:3]
    ex_txt = "\n".join(f"- “{e['sentence'][:110]}” → **{e['response']}** (label: {e['label']})" for e in ex)
    return "\n\n".join([
        f"{r['model']}, {r['data']['train']:,} training examples, {r['train_seconds'] / 60:.0f} min. {lora_txt}{peak}",
        "Same PhraseBank test set as Ch 6, answered in free text:",
        table(["Model", "Accuracy", "Macro-F1", "Unparseable answers"], rows), ex_txt,
        "Free-form finance answers are in [`results/ch07_responses.json`](../results/ch07_responses.json).",
        img(r["figures"][0], "instruction tuning loss")])


def sec_capstone(sc, bt):
    if not bt:
        return "_Not run yet: `python scripts/capstone_score_news.py` then `python scripts/capstone_backtest.py`_"
    rows = []
    for m, per in bt["results"].items():
        for p, r in per.items():
            ic1 = r["ic"]["open_t_to_open_t+1"]
            ic0 = r["ic"]["same_day_close_to_close (not tradable)"]
            rows.append([m, p.replace("_", "-"), f(r["net"]["sharpe"], 2) + " " + ci(r["net_sharpe_ci"], 2),
                         f(r["gross"]["sharpe"], 2), f(r["net"]["ann_return"], pct=True),
                         f(r["net"]["max_drawdown"], pct=True),
                         f"{ic1['mean_ic']:+.4f} (t={ic1['nw_tstat']:.1f})",
                         f"{ic0['mean_ic']:+.4f} (t={ic0['nw_tstat']:.1f})",
                         f(r["permutation"]["p_value_one_sided"], 3), f(r["avg_daily_turnover"], 2)])
    pr = bt["periods"]
    cov = bt["coverage"]
    head = (f"In-sample {pr['pretrain_end']} → {pr['is_end']}, out-of-sample {pr['is_end']} → {pr['end']}. "
            f"{sc['n_headlines']:,} headlines on {sc['universe_size']} stocks scored with {sc['model']} "
            f"({sc['headlines_per_second']:,.0f} headlines/s). Price coverage {cov['headlines_with_prices']:.0%} of headlines. "
            if sc else "")
    s = bt["settings"]
    return "\n\n".join([
        head + f"Rules: enter at the open after the news date, hold one day, ±{s['threshold']} score threshold, "
        f"≥{s['min_names']} names per side, price ≥ ${s['min_price']:g}, {s['cost_bps']:g} bps per side.",
        table(["Signal", "Period", "Net Sharpe [95% CI]", "Gross Sharpe", "Net ann. return", "Max DD",
               "IC next day (NW t)", "IC same day (NW t)", "Permutation p", "Daily turnover"], rows),
        "**Caveats.** " + " ".join(bt["caveats"]),
        "\n\n".join(img(p, "capstone") for p in bt["figures"])])


def build(res_dir: Path):
    r = {n: load(res_dir, n + ".json") for n in
         ["ch02_tokenization", "ch03_attention", "ch04_model", "ch05_pretrain", "ch05_gpt2_domain", "ch05_dapt",
          "ch06_sentiment", "ch07_instruct", "capstone_scoring", "capstone_backtest"]}
    full = "\n\n".join([
        "# Results\n\n_Generated by `scripts/make_report.py` from the JSON files in `results/`. Do not edit by hand._",
        "## Ch 2 — Tokenising financial text\n\n" + sec_ch2(r["ch02_tokenization"]),
        "## Ch 3 — Attention\n\n" + sec_ch3(r["ch03_attention"]),
        "## Ch 4 — Model size and what fits on the GPU\n\n" + sec_ch4(r["ch04_model"]),
        "## Ch 5 — Pretraining, GPT-2 weights, domain adaptation\n\n" + sec_ch5(r["ch05_pretrain"], r["ch05_gpt2_domain"], r["ch05_dapt"]),
        "## Ch 6 — Financial sentiment classifier\n\n" + sec_ch6(r["ch06_sentiment"]),
        "## Ch 7 — Instruction-tuned finance assistant\n\n" + sec_ch7(r["ch07_instruct"], r["ch06_sentiment"]),
        "## Capstone — Sentiment as a trading signal\n\n" + sec_capstone(r["capstone_scoring"], r["capstone_backtest"]),
    ]) + "\n"
    return r, full


def readme_block(r):
    lines = []
    c6, c7, d, bt = r["ch06_sentiment"], r["ch07_instruct"], r["ch05_dapt"], r["capstone_backtest"]
    if c6:
        best = c6["models"][c6["selected_by_validation"]]
        tf = c6["models"]["TF-IDF + LogReg"]["test"]
        diff = best.get("vs_tfidf_macro_f1", {})
        lines.append(f"- **Sentiment (Ch 6):** {c6['selected_by_validation']} reaches macro-F1 "
                     f"**{best['test']['macro_f1']:.3f}** {ci(best['test']['macro_f1_ci'])} on held-out PhraseBank, "
                     f"vs {tf['macro_f1']:.3f} for TF-IDF + logistic regression "
                     f"(Δ {diff.get('diff', float('nan')):+.3f}, McNemar p = {best.get('vs_tfidf_mcnemar', {}).get('p_value', float('nan')):.4f}).")
    if d:
        k = [c for c in d["perplexity_before"] if "out-of-sample" in c][0]
        g = [c for c in d["perplexity_before"] if "General" in c][0]
        ch = lambda c: d["perplexity_after"][c]["perplexity"] / d["perplexity_before"][c]["perplexity"] - 1
        lines.append(f"- **Domain adaptation (Ch 5):** continued pretraining cut GPT-2's perplexity on unseen "
                     f"financial headlines by **{-ch(k):.0%}**, at a cost of {ch(g):+.0%} on general English.")
    if c7:
        a = c7["sentiment_after"]
        lines.append(f"- **Instruction tuning (Ch 7):** {c7['model']}" + (f" + LoRA r={c7['lora']['rank']}" if c7.get("lora") else "")
                     + f" answers sentiment questions with accuracy **{a['accuracy']:.3f}** "
                     f"(zero-shot base model: {c7['sentiment_before']['accuracy']:.3f}).")
    if bt:
        m = next(iter(bt["results"]))
        o = bt["results"][m]["out_of_sample"]
        lines.append(f"- **Trading signal (capstone):** out-of-sample, next-open long/short on {m.lower()} sentiment: "
                     f"net Sharpe **{o['net']['sharpe']:.2f}** {ci(o['net_sharpe_ci'], 2)} after "
                     f"{bt['settings']['cost_bps']:g} bps costs (gross {o['gross']['sharpe']:.2f}); next-day IC "
                     f"{o['ic']['open_t_to_open_t+1']['mean_ic']:+.4f}, same-day IC "
                     f"{o['ic']['same_day_close_to_close (not tradable)']['mean_ic']:+.4f}; permutation p = "
                     f"{o['permutation']['p_value_one_sided']:.3f}.")
    if not lines:
        lines.append("_No results yet. Run `.\\run_all.ps1` (Windows) to produce them; this block fills in automatically._")
    return "\n".join(lines) + "\n\nFull tables and figures: [reports/RESULTS.md](reports/RESULTS.md)."


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="build from results/smoke (output to results/smoke/RESULTS.md)")
    args = p.parse_args()
    res_dir = ROOT / "results" / ("smoke" if args.smoke else "")
    r, full = build(res_dir)
    if args.smoke:
        out = res_dir / "RESULTS.md"
        out.write_text(full.replace("](../results/smoke/", "](../../results/smoke/"), encoding="utf-8")
        print("wrote", out)
        return
    out = ROOT / "reports" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(full, encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    block = f"{START}\n{readme_block(r)}\n{END}"
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    readme.write_text(text, encoding="utf-8")
    print("wrote", out, "and updated README results block")


if __name__ == "__main__":
    main()
