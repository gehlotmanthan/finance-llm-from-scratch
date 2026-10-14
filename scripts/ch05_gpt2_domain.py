"""Chapter 5.5 — load OpenAI's GPT-2 into our own GPTModel and measure the
"domain gap" between general English and financial text.

1. Loads GPT-2 weights (Hugging Face safetensors) into finllm.model.GPTModel and
   checks it produces sensible text (greedy and temperature/top-k sampling).
2. Perplexity of GPT-2 on: general English (The Verdict), Financial PhraseBank
   (test split) and news headlines from the *out-of-sample* window.
3. Attention inside the pretrained model on a headline: the share of attention
   each layer puts on the first token (the well-known "attention sink") and the
   pattern of the head that looks most at the content words.

Output: results/ch05_gpt2_domain.json + 2 figures.
"""

import numpy as np
import torch

from _common import base_parser, load_base_gpt2, news_and_periods, plt, rel, savefig, SERIES, SEQ_BLUES, INK_2
from finllm import data_sources as ds
from finllm import synthetic
from finllm.data import join_documents
from finllm.generate import generate, text_to_token_ids, token_ids_to_text
from finllm.model import get_attention_maps, set_attention_capture
from finllm.paths import results_dirs
from finllm.tokenizer import EOS_ID, get_gpt2_tokenizer
from finllm.train import corpus_loss
from finllm.utils import device_name, get_device, run_metadata, save_json, set_seed
from matplotlib.colors import LinearSegmentedColormap

PROMPTS = ["Shares of Apple rose after", "The Federal Reserve said on Wednesday that",
           "Quarterly earnings per share came in"]
HEADLINE = "Tesla shares fall after the company misses quarterly delivery estimates"


def eval_corpora(args, tok, news, periods):
    verdict = ds.load_the_verdict()
    if args.smoke:
        pb_test = synthetic.phrasebank_like(200)["text"].tolist()
    else:
        pb_test = ds.phrasebank_splits(seed=args.seed)["test"]["text"].tolist()
    oos = ds.headlines_corpus(news, periods["end"], 400_000 if not args.smoke else 20_000,
                              seed=args.seed, start_date=periods["is_end"])
    return {"General English (The Verdict)": verdict,
            "Financial PhraseBank (test)": join_documents(pb_test),
            "News headlines (out-of-sample window)": join_documents(oos)}


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=["gpt2-small (124M)", "gpt2-medium (355M)"])
    p.add_argument("--max-windows", type=int, default=400)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, figs = results_dirs(args.smoke)
    tok = get_gpt2_tokenizer()
    news, periods = news_and_periods(args.smoke)
    corpora = eval_corpora(args, tok, news, periods)
    max_len = 32 if args.smoke else 256

    ppl, samples = {}, {}
    attention = None
    for name in (args.models[:1] if args.smoke else args.models):
        model, cfg = load_base_gpt2(name, device, args.smoke, use_sdpa=True)
        ppl[name] = {}
        for cname, text in corpora.items():
            r = corpus_loss(model, text, tok, device, max_length=max_len, max_windows=args.max_windows)
            ppl[name][cname] = r
            print(f"{name:20s} {cname:40s} ppl {r['perplexity']:8.1f}  ({r['tokens']:,} tokens)")

        samples[name] = []
        for prompt in PROMPTS:
            ids = text_to_token_ids(prompt, tok).to(device)
            greedy = token_ids_to_text(generate(model, ids, 25, cfg["context_length"]).cpu(), tok)
            torch.manual_seed(args.seed)
            sampled = token_ids_to_text(generate(model, ids, 25, cfg["context_length"], temperature=0.8,
                                                 top_k=40, eos_id=EOS_ID).cpu(), tok)
            samples[name].append({"prompt": prompt, "greedy": greedy, "temp0.8_top40": sampled})
            print("  greedy :", repr(greedy))

        if attention is None:  # attention analysis on the first (smallest) model
            set_attention_capture(model, True)
            ids = text_to_token_ids(HEADLINE, tok).to(device)
            with torch.no_grad():
                model(ids)
            maps = [m[0].float().cpu().numpy() for m in get_attention_maps(model)]  # each (h, T, T)
            set_attention_capture(model, False)
            pieces = [tok.decode([i]) for i in ids[0].tolist()]
            sink = [float(m[:, 1:, 0].mean()) for m in maps]          # attention on token 0, queries 1..T-1
            content = [i for i, s in enumerate(pieces) if s.strip().lower() in {"fall", "misses", "estimates"}]
            best = max(((l, h) for l in range(1, len(maps)) for h in range(maps[l].shape[0])),
                       key=lambda lh: maps[lh[0]][lh[1]][:, content].sum() if content else -maps[lh[0]][lh[1]][1:, 0].mean())
            attention = {"model": name, "headline": HEADLINE, "tokens": pieces,
                         "sink_share_by_layer": sink, "head_shown": {"layer": best[0], "head": best[1]},
                         "head_matrix": maps[best[0]][best[1]].round(4).tolist()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Figure 1: perplexity by corpus and model
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    cnames = list(corpora)
    width = 0.8 / len(ppl)
    for i, (mname, d) in enumerate(ppl.items()):
        ys = [d[c]["perplexity"] for c in cnames]
        xs = np.arange(len(cnames)) + i * width
        ax.bar(xs, ys, width=width * 0.92, color=SERIES[i], label=mname)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(len(cnames)) + width * (len(ppl) - 1) / 2, [c.replace(" (", "\n(") for c in cnames])
    ax.set(ylabel="perplexity (lower = less surprised)", title="How surprised is GPT-2 by each kind of text?")
    ax.grid(axis="x", visible=False)
    ax.legend()
    savefig(fig, figs, "ch05_domain_perplexity.png")

    # Figure 2: attention
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    a1.bar(range(len(attention["sink_share_by_layer"])), attention["sink_share_by_layer"], color=SERIES[0])
    a1.set(xlabel="layer", ylabel="mean attention on first token", title="Attention sink by layer")
    a1.set_xticks(range(len(attention["sink_share_by_layer"])))
    a1.grid(axis="x", visible=False)
    mat = np.array(attention["head_matrix"])
    cmap = LinearSegmentedColormap.from_list("blues", ["#fcfcfb"] + SEQ_BLUES)
    a2.imshow(mat, cmap=cmap, vmin=0, vmax=max(mat[:, 1:].max(), 1e-6))
    labels = [t.strip() or "␣" for t in attention["tokens"]]
    a2.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    a2.set_yticks(range(len(labels)), labels, fontsize=8)
    a2.grid(False)
    a2.set_title(f"Layer {attention['head_shown']['layer']}, head {attention['head_shown']['head']}: "
                 "rows attend to columns", fontsize=10)
    savefig(fig, figs, "ch05_attention.png")

    result = {"chapter": 5, "part": "gpt2_domain_gap", "smoke": args.smoke, "device": device_name(device),
              "eval_window_tokens": max_len, "perplexity": ppl, "samples": samples, "attention": attention,
              "periods": {k: str(v.date()) for k, v in periods.items()},
              "figures": [rel(figs / "ch05_domain_perplexity.png"), rel(figs / "ch05_attention.png")],
              "meta": run_metadata(device, seed=args.seed, args=vars(args))}
    print("saved", rel(save_json(result, out_dir / "ch05_gpt2_domain.json")))


if __name__ == "__main__":
    main()
