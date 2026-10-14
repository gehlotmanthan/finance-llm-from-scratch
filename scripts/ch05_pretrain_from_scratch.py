"""Chapter 5 — pretrain a small GPT from scratch on financial news headlines.

Model: FIN_SMALL_CONFIG (6 layers, 384 wide, 128-token context, ~30M params).
Data: headlines dated inside the *pretraining window only* (the first 40% of
the news calendar), so nothing the model reads overlaps the backtest periods.

Reported: train/val loss curves, validation perplexity, a unigram
(word-frequency) baseline the model has to beat, and generated samples.

Output: results/ch05_pretrain.json, figure, checkpoints/fin-small-scratch.pth
"""

import torch

from _common import base_parser, news_and_periods, plt, rel, savefig, SERIES, INK_2
from finllm import data_sources as ds
from finllm.config import FIN_SMALL_CONFIG, TINY_CONFIG
from finllm.data import create_dataloader_v1, join_documents, train_val_split_text
from finllm.generate import generate, text_to_token_ids, token_ids_to_text
from finllm.model import GPTModel, count_parameters
from finllm.paths import checkpoint_path, results_dirs
from finllm.tokenizer import EOS_ID, EOS_TOKEN, get_gpt2_tokenizer
from finllm.train import perplexity, save_checkpoint, train_lm, unigram_baseline_loss
from finllm.utils import Timer, device_name, get_device, run_metadata, save_json, set_seed

PROMPTS = ["Apple shares", "The Federal Reserve", "Oil prices", "Tesla stock falls"]


def main():
    p = base_parser(__doc__)
    p.add_argument("--max-chars", type=int, default=20_000_000, help="size of the training text (characters)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, figs = results_dirs(args.smoke)
    tok = get_gpt2_tokenizer()

    news, periods = news_and_periods(args.smoke)
    max_chars = 60_000 if args.smoke else args.max_chars
    heads = ds.headlines_corpus(news, periods["pretrain_end"], max_chars, seed=args.seed)
    train_txt, val_txt = train_val_split_text(join_documents(heads), 0.9)
    cfg = dict(TINY_CONFIG, context_length=32) if args.smoke else dict(FIN_SMALL_CONFIG)
    ctx = cfg["context_length"]
    train_dl = create_dataloader_v1(train_txt, args.batch_size, ctx, ctx, shuffle=True, drop_last=True)
    val_dl = create_dataloader_v1(val_txt, args.batch_size, ctx, ctx, shuffle=False, drop_last=False)
    print(f"{len(heads):,} headlines up to {periods['pretrain_end'].date()} | "
          f"{len(train_dl.dataset):,} train / {len(val_dl.dataset):,} val windows of {ctx} tokens")

    torch.manual_seed(args.seed)
    model = GPTModel(cfg).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    steps = len(train_dl) * (1 if args.smoke else args.epochs)
    with Timer() as t:
        hist = train_lm(model, train_dl, val_dl, opt, device,
                        num_epochs=1 if args.smoke else args.epochs,
                        eval_freq=max(steps // 20, 1), eval_iter=20,
                        start_context=PROMPTS[0], tokenizer=tok,
                        max_lr=args.lr, warmup_steps=max(steps // 20, 1), min_lr=args.lr / 10,
                        grad_clip=1.0, amp=True)

    train_ids = tok.encode(train_txt, allowed_special={EOS_TOKEN})
    val_ids = tok.encode(val_txt, allowed_special={EOS_TOKEN})
    unigram = unigram_baseline_loss(train_ids, val_ids)
    final_val = hist["val_loss"][-1]
    print(f"final val loss {final_val:.3f} (ppl {perplexity(final_val):.1f}) vs unigram baseline "
          f"{unigram:.3f} (ppl {perplexity(unigram):.1f})")

    samples = []
    model.eval()
    for prompt in PROMPTS:
        torch.manual_seed(args.seed)
        ids = generate(model, text_to_token_ids(prompt, tok).to(device), 20, ctx,
                       temperature=0.8, top_k=40, eos_id=EOS_ID)
        samples.append({"prompt": prompt, "text": token_ids_to_text(ids.cpu(), tok)})
        print(" ", samples[-1]["text"])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    toks_m = [t / 1e6 for t in hist["tokens_seen"]]
    ax.plot(toks_m, hist["train_loss"], color=SERIES[0], label="train")
    ax.plot(toks_m, hist["val_loss"], color=SERIES[1], label="validation")
    ax.axhline(unigram, color=INK_2, lw=1, ls="--")
    ax.text(toks_m[0], unigram, " word-frequency baseline", va="bottom", fontsize=8, color=INK_2)
    ax.set(xlabel="tokens seen (millions)", ylabel="cross-entropy loss",
           title=f"Pretraining a {n_params / 1e6:.0f}M-parameter GPT on financial headlines")
    ax.legend()
    savefig(fig, figs, "ch05_pretrain_loss.png")

    ckpt = save_checkpoint(checkpoint_path("fin-small-scratch.pth", args.smoke), model, cfg)
    result = {"chapter": 5, "part": "pretrain_from_scratch", "smoke": args.smoke,
              "device": device_name(device), "config": cfg, "params": n_params,
              "data": {"headlines": len(heads), "train_tokens": len(train_ids), "val_tokens": len(val_ids),
                       "window_end": str(periods["pretrain_end"].date())},
              "train_seconds": t.seconds, "history": hist,
              "final_val_loss": final_val, "final_val_ppl": perplexity(final_val),
              "unigram_val_loss": unigram, "unigram_val_ppl": perplexity(unigram),
              "samples": samples, "checkpoint": rel(ckpt),
              "figures": [rel(figs / "ch05_pretrain_loss.png")],
              "meta": run_metadata(device, seed=args.seed, args=vars(args))}
    print("saved", rel(save_json(result, out_dir / "ch05_pretrain.json")))


if __name__ == "__main__":
    main()
