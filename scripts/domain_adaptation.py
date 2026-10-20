"""Domain-adaptive pretraining (DAPT).

Continue GPT-2's next-token training on financial headlines from the
pretraining window (Gururangan et al., 2020, "Don't Stop Pretraining"), then
measure:

  * perplexity on held-out *out-of-sample* headlines (does it learn the domain?)
  * perplexity on Financial PhraseBank (does it transfer to other finance text?)
  * perplexity on general English (how much does it forget?)

The adapted weights are the "dapt" starting point compared against vanilla
GPT-2 in sentiment_classifier.py (does domain pretraining improve sentiment classification?).

Output: results/domain_adaptation.json, figure, checkpoints/gpt2-small-dapt.pth
"""

import torch

from _common import base_parser, load_base_gpt2, news_and_periods, plt, rel, savefig, SERIES
from domain_gap import eval_corpora
from finllm import data_sources as ds
from finllm.data import create_dataloader_v1, join_documents, train_val_split_text
from finllm.paths import checkpoint_path, results_dirs
from finllm.tokenizer import get_gpt2_tokenizer
from finllm.train import corpus_loss, save_checkpoint, train_lm
from finllm.utils import Timer, device_name, get_device, run_metadata, save_json, set_seed


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", default="gpt2-small (124M)")
    p.add_argument("--max-chars", type=int, default=25_000_000)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-windows", type=int, default=400)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, figs = results_dirs(args.smoke)
    tok = get_gpt2_tokenizer()
    news, periods = news_and_periods(args.smoke)
    corpora = eval_corpora(args, tok, news, periods)
    ctx = 32 if args.smoke else args.context
    max_steps = 10 if args.smoke else args.max_steps

    model, cfg = load_base_gpt2(args.model, device, args.smoke, use_sdpa=True, drop_rate=0.0)

    def evaluate(tag):
        out = {}
        for cname, text in corpora.items():
            out[cname] = corpus_loss(model, text, tok, device, max_length=ctx, max_windows=args.max_windows)
            print(f"[{tag}] {cname:40s} ppl {out[cname]['perplexity']:.1f}")
        return out

    before = evaluate("before")

    heads = ds.headlines_corpus(news, periods["pretrain_end"], 60_000 if args.smoke else args.max_chars,
                                seed=args.seed)
    train_txt, val_txt = train_val_split_text(join_documents(heads), 0.98)
    train_dl = create_dataloader_v1(train_txt, args.batch_size, ctx, ctx, shuffle=True)
    val_dl = create_dataloader_v1(val_txt, args.batch_size, ctx, ctx, shuffle=False, drop_last=False)
    print(f"DAPT corpus: {len(heads):,} headlines <= {periods['pretrain_end'].date()}, "
          f"{len(train_dl.dataset):,} windows; training {max_steps} steps")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    with Timer() as t:
        hist = train_lm(model, train_dl, val_dl, opt, device, num_epochs=100, eval_freq=max(max_steps // 15, 1),
                        eval_iter=20, max_lr=args.lr, warmup_steps=max(max_steps // 20, 1),
                        min_lr=args.lr / 10, grad_clip=1.0, grad_accum=args.grad_accum, amp=True,
                        max_steps=max_steps)
    after = evaluate("after")

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    names = list(corpora)
    import numpy as np
    xs = np.arange(len(names))
    b = [before[n]["perplexity"] for n in names]
    a = [after[n]["perplexity"] for n in names]
    ax.bar(xs - 0.2, b, 0.38, color=SERIES[0], label="GPT-2 (OpenAI weights)")
    ax.bar(xs + 0.2, a, 0.38, color=SERIES[1], label="GPT-2 + finance DAPT")
    for x, y0, y1 in zip(xs, b, a):
        ax.text(x + 0.2, y1, f"{(y1 / y0 - 1):+.0%}", ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(xs, [n.replace(" (", "\n(") for n in names])
    ax.set(ylabel="perplexity", title="Domain-adaptive pretraining: gain vs forgetting")
    ax.grid(axis="x", visible=False)
    ax.legend()
    savefig(fig, figs, "domain_adaptation.png")

    ckpt = save_checkpoint(checkpoint_path("gpt2-small-dapt.pth", args.smoke), model, cfg,
                           base_model=args.model, steps=hist["total_steps"])
    result = {"experiment": "domain_adaptation", "smoke": args.smoke, "device": device_name(device),
              "base_model": args.model, "train_seconds": t.seconds, "tokens_trained": hist["total_tokens"],
              "data": {"headlines": len(heads), "window_end": str(periods["pretrain_end"].date())},
              "perplexity_before": before, "perplexity_after": after, "history": hist,
              "checkpoint": rel(ckpt), "figures": [rel(figs / "domain_adaptation.png")],
              "meta": run_metadata(device, seed=args.seed, args=vars(args))}
    print("saved", rel(save_json(result, out_dir / "domain_adaptation.json")))


if __name__ == "__main__":
    main()
