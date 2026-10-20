"""How well does GPT-2's tokenizer fit financial text?

Measures, on real data:
  * BPE efficiency (tokens per word, share of words split) on financial news
    headlines and PhraseBank sentences vs general English (The Verdict);
  * how finance-specific terms and tickers get fragmented;
  * why a closed word-level vocabulary fails (out-of-vocabulary rate);
  * the sliding-window dataloader's output on the finance corpus.

Output: results/tokenizer_study.json + 2 figures.
"""

from collections import Counter

from _common import base_parser, plt, rel, savefig, SERIES, INK_2
from finllm import data_sources as ds
from finllm import synthetic
from finllm.data import create_dataloader_v1, join_documents
from finllm.paths import results_dirs
from finllm.tokenizer import (bpe_stats, get_gpt2_tokenizer, word_level_oov_rate, word_pieces)
from finllm.utils import run_metadata, save_json, set_seed

FINANCE_TERMS = ["EBITDA", "EPS", "y/y", "bps", "guidance", "dividend", "buyback", "downgrade",
                 "AAPL", "NVDA", "BRK.B", "Q3", "10-K", "FY2026", "$1.2bn", "+3.5%",
                 "FOMC", "hawkish", "SOFR", "ETF"]


def main():
    args = base_parser(__doc__).parse_args()
    set_seed(args.seed)
    out_dir, figs = results_dirs(args.smoke)
    enc = get_gpt2_tokenizer()

    verdict = ds.load_the_verdict()
    if args.smoke:
        pb = synthetic.phrasebank_like(300)
        news = synthetic.news_like(60, per_day=10)
    else:
        pb = ds.load_phrasebank()
        news = ds.load_news()
    heads = news["headline"].sample(n=min(20000, len(news)), random_state=args.seed).tolist()

    corpora = {
        "General English (The Verdict)": [verdict],
        "PhraseBank sentences": pb["text"].tolist(),
        "News headlines": heads,
    }
    stats = {name: bpe_stats(texts, enc) for name, texts in corpora.items()}
    for name, s in stats.items():
        print(f"{name:32s} tokens/word {s['tokens_per_word']:.2f}  "
              f"chars/token {s['chars_per_token']:.2f}  split words {s['share_words_split']:.1%}")

    terms = {t: word_pieces(t, enc) for t in FINANCE_TERMS}

    # Most frequent headline words that BPE breaks into 3+ pieces
    counts = Counter(w for h in heads for w in h.split())
    frag = [(w, c, len(enc.encode(" " + w))) for w, c in counts.most_common(5000)]
    worst = [{"word": w, "count": c, "tokens": k, "pieces": word_pieces(w, enc)}
             for w, c, k in frag if k >= 3 and w.isalpha()][:15]

    # Closed-vocabulary failure
    oov = {
        "vocab_from_verdict_on_phrasebank": word_level_oov_rate([verdict], pb["text"].tolist()),
        "vocab_from_phrasebank_on_headlines": word_level_oov_rate(pb["text"].tolist(), heads),
    }
    for k, v in oov.items():
        print(f"OOV {k}: {v['oov_rate']:.1%} of tokens become <|unk|> (vocab {v['vocab_size']:,})")

    # Sliding-window dataloader on the finance corpus
    corpus = join_documents(heads[:5000])
    dl = create_dataloader_v1(corpus, batch_size=8, max_length=32, stride=32, shuffle=False)
    x, y = next(iter(dl))
    loader = {"max_length": 32, "stride": 32, "batch_size": 8, "n_windows": len(dl.dataset),
              "batch_shape": list(x.shape),
              "first_input_text": enc.decode(x[0].tolist()),
              "first_target_text": enc.decode(y[0].tolist()),
              "targets_are_inputs_shifted_by_one": bool((x[0, 1:] == y[0, :-1]).all())}

    # Figures
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    names = list(stats)
    vals = [stats[n]["tokens_per_word"] for n in names]
    ax.barh(names, vals, color=SERIES[0], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=INK_2)
    ax.set_xlabel("GPT-2 BPE tokens per word (lower = more efficient)")
    ax.set_title("GPT-2 tokenizer efficiency by corpus")
    ax.grid(axis="y", visible=False)
    savefig(fig, figs, "tokenizer_tokens_per_word.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    items = sorted(terms.items(), key=lambda kv: len(kv[1]))
    ax.barh([t for t, _ in items], [len(p) for _, p in items], color=SERIES[0], height=0.6)
    for i, (t, p) in enumerate(items):
        ax.text(len(p) + 0.05, i, " | ".join(s.strip() or "␣" for s in p), va="center",
                fontsize=8, color=INK_2)
    ax.set_xlabel("number of BPE tokens")
    ax.set_title("How GPT-2 splits finance vocabulary")
    ax.set_xlim(0, max(len(p) for p in terms.values()) + 3)
    ax.grid(axis="y", visible=False)
    savefig(fig, figs, "tokenizer_term_fragmentation.png")

    result = {
        "experiment": "tokenizer_study", "smoke": args.smoke,
        "bpe_stats": stats, "finance_terms": terms, "most_fragmented_headline_words": worst,
        "word_level_oov": oov, "dataloader": loader,
        "figures": [rel(figs / "tokenizer_tokens_per_word.png"), rel(figs / "tokenizer_term_fragmentation.png")],
        "meta": run_metadata(seed=args.seed),
    }
    print("saved", rel(save_json(result, out_dir / "tokenizer_study.json")))


if __name__ == "__main__":
    main()
