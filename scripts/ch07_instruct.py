"""Chapter 7 — instruction fine-tuning GPT-2 into a small finance assistant.

Training data
  * finance-alpaca (general finance Q&A, MIT licence), examples <= 256 tokens
  * sentiment instructions built from the Ch 6 PhraseBank TRAINING split

Evaluation
  * objective: the instruction-tuned model answers "positive/negative/neutral"
    for the Ch 6 PhraseBank TEST sentences -> accuracy and macro-F1 with 95% CIs,
    before and after tuning, comparable with the Ch 6 classifier
  * qualitative: answers to held-out finance-alpaca questions (saved for review
    and for the optional Ollama judge, scripts/ch07_ollama_judge.py)

Default model: GPT-2 medium (355M) with LoRA rank 16 (Appendix E) so it fits on
an 8 GB GPU; ``--lora-rank 0`` does full fine-tuning (book setting).

Output: results/ch07_instruct.json, results/ch07_responses.json, figure, checkpoint
"""

import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from _common import base_parser, load_base_gpt2, plt, rel, savefig, SERIES, INK_2
from finllm import data_sources as ds
from finllm import synthetic
from finllm.generate import generate, text_to_token_ids, token_ids_to_text
from finllm.instruct import (InstructionDataset, custom_collate_fn, extract_response, format_input,
                             parse_label, response_prefix, sentiment_instructions, token_length)
from finllm.lora import apply_lora
from finllm.paths import checkpoint_path, results_dirs
from finllm.stats import accuracy, bootstrap_ci, classification_metrics, macro_f1
from finllm.tokenizer import EOS_ID, get_gpt2_tokenizer
from finllm.train import calc_loss_loader, train_lm
from finllm.utils import Timer, device_name, get_device, run_metadata, save_json, set_seed


@torch.no_grad()
def respond(model, tok, entry, device, ctx, max_new_tokens):
    prompt = format_input(entry) + response_prefix()
    ids = text_to_token_ids(prompt, tok).to(device)
    out = generate(model, ids, max_new_tokens, ctx, eos_id=EOS_ID)
    return extract_response(token_ids_to_text(out.cpu(), tok), prompt)


def sentiment_eval(model, tok, test_df, device, ctx, seed):
    model.eval()
    labels = ds.LABELS
    preds, raw = [], []
    for e in sentiment_instructions(test_df):
        r = respond(model, tok, e, device, ctx, max_new_tokens=4)
        raw.append(r)
        preds.append(parse_label(r))
    y = test_df["label_id"].to_numpy()
    valid = np.array([p is not None for p in preds])
    # invalid answers count as wrong: map them to an impossible class id
    pred_ids = np.array([ds.LABEL2ID[p] if p else -1 for p in preds])
    pid = np.where(pred_ids < 0, 3, pred_ids)
    f1 = lambda a, b: macro_f1(a, np.where(b == 3, (a + 1) % 3, b), 3)   # invalid -> scored as a wrong class
    m = classification_metrics(y, np.where(pid == 3, (y + 1) % 3, pid), labels)
    m["accuracy_ci"] = bootstrap_ci(accuracy, y, pid, seed=seed)
    m["macro_f1_ci"] = bootstrap_ci(f1, y, pid, seed=seed)
    m["invalid_rate"] = float(1 - valid.mean())
    m["examples"] = [{"sentence": t, "label": l, "response": r}
                     for t, l, r in list(zip(test_df["text"], test_df["label"], raw))[:8]]
    return m


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", default="gpt2-medium (355M)")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16)
    p.add_argument("--n-alpaca", type=int, default=6000)
    p.add_argument("--n-sentiment", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch-size x grad-accum")
    p.add_argument("--lr", type=float, default=None, help="default 1e-4 with LoRA, 5e-5 full fine-tune")
    p.add_argument("--mask-prompt", action="store_true", help="Exercise 7.2: no loss on prompt tokens")
    p.add_argument("--n-sample-responses", type=int, default=25)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, figs = results_dirs(args.smoke)
    tok = get_gpt2_tokenizer()

    # ---------------- data
    if args.smoke:
        alpaca = synthetic.instruction_like(60)
        splits = ds.split_phrasebank(synthetic.phrasebank_like(200), seed=args.seed)
        n_alpaca, n_sent, max_tokens = 50, 40, 96
    else:
        alpaca = ds.load_finance_alpaca()
        splits = ds.phrasebank_splits(seed=args.seed)
        n_alpaca, n_sent, max_tokens = args.n_alpaca, args.n_sentiment, args.max_tokens
    rng = random.Random(args.seed)
    alpaca = [e for e in alpaca if token_length(e, tok) <= max_tokens]
    rng.shuffle(alpaca)
    n_test = max(5, min(args.n_sample_responses, len(alpaca) // 10))
    a_test, a_val, a_train = alpaca[:n_test], alpaca[n_test:n_test + max(10, n_alpaca // 20)], \
        alpaca[n_test + max(10, n_alpaca // 20):][:n_alpaca]
    s_train = sentiment_instructions(splits["train"].sample(n=min(n_sent, len(splits["train"])), random_state=args.seed))
    s_val = sentiment_instructions(splits["val"])
    train_data = a_train + s_train
    val_data = a_val + s_val[:len(a_val)]
    rng.shuffle(train_data)
    print(f"train {len(train_data):,} ({len(a_train):,} finance Q&A + {len(s_train):,} sentiment) | "
          f"val {len(val_data):,} | sentiment test {len(splits['test']):,}")

    collate = partial(custom_collate_fn, allowed_max_length=max_tokens, mask_prompt=args.mask_prompt)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(InstructionDataset(train_data, tok), args.batch_size, shuffle=True, drop_last=True,
                          collate_fn=collate, generator=g)
    val_dl = DataLoader(InstructionDataset(val_data, tok), args.batch_size, collate_fn=collate)

    # ---------------- model
    model, cfg = load_base_gpt2(args.model, device, args.smoke, use_sdpa=True, drop_rate=0.0)
    ctx = cfg["context_length"]
    test_df = splits["test"]
    print("zero-shot sentiment eval of the base model ...")
    before = sentiment_eval(model, tok, test_df, device, ctx, args.seed)
    print(f"  before: acc {before['accuracy']:.3f}, invalid answers {before['invalid_rate']:.1%}")
    base_val_loss = calc_loss_loader(val_dl, model, device)

    lora_info = None
    if args.lora_rank > 0:
        trainable, total = apply_lora(model, args.lora_rank, args.lora_alpha)
        model.to(device)
        lora_info = {"rank": args.lora_rank, "alpha": args.lora_alpha, "trainable_params": trainable,
                     "total_params": total, "trainable_share": trainable / total}
        print(f"LoRA: {trainable:,} trainable of {total:,} parameters ({trainable / total:.2%})")
    lr = args.lr or (1e-4 if args.lora_rank > 0 else 5e-5)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.1)
    epochs = 1 if args.smoke else args.epochs
    steps = len(train_dl) * epochs // args.grad_accum
    model.train()
    with Timer() as t:
        hist = train_lm(model, train_dl, val_dl, opt, device, num_epochs=epochs,
                        eval_freq=max(steps // 12, 1), eval_iter=10, max_lr=lr,
                        warmup_steps=max(steps // 20, 1), min_lr=lr / 10, grad_clip=1.0, grad_accum=args.grad_accum, amp=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None

    print("sentiment eval after instruction tuning ...")
    after = sentiment_eval(model, tok, test_df, device, ctx, args.seed)
    print(f"  after: acc {after['accuracy']:.3f}  macro-F1 {after['macro_f1']:.3f}  "
          f"invalid {after['invalid_rate']:.1%}")

    responses = []
    model.eval()
    for e in a_test:
        responses.append({"instruction": e["instruction"], "input": e["input"], "reference": e["output"],
                          "model_response": respond(model, tok, e, device, ctx, max_new_tokens=120)})
    print("example:", responses[0]["instruction"], "->", responses[0]["model_response"][:200])

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    tk = [x / 1e6 for x in hist["tokens_seen"]]
    ax.plot(tk, hist["train_loss"], color=SERIES[0], label="train")
    ax.plot(tk, hist["val_loss"], color=SERIES[1], label="validation")
    ax.axhline(base_val_loss, color=INK_2, lw=1, ls="--")
    ax.text(tk[0], base_val_loss, " base model (before tuning)", va="bottom", fontsize=8, color=INK_2)
    ax.set(xlabel="tokens seen (millions)", ylabel="loss",
           title=f"Instruction tuning {args.model}" + (f" with LoRA r={args.lora_rank}" if lora_info else ""))
    ax.legend()
    savefig(fig, figs, "ch07_loss.png")

    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "lora": lora_info, "base_model": args.model},
               checkpoint_path("instruct.pth", args.smoke))
    save_json({"model": args.model, "responses": responses}, out_dir / "ch07_responses.json")
    result = {"chapter": 7, "smoke": args.smoke, "device": device_name(device), "model": args.model,
              "lora": lora_info, "lr": lr, "epochs": epochs,
              "batch_size": args.batch_size, "grad_accum": args.grad_accum, "mask_prompt": args.mask_prompt,
              "data": {"train": len(train_data), "train_alpaca": len(a_train), "train_sentiment": len(s_train),
                       "val": len(val_data), "max_tokens": max_tokens},
              "train_seconds": t.seconds, "peak_gpu_gb": peak_gb, "base_val_loss": base_val_loss,
              "history": hist, "sentiment_before": before, "sentiment_after": after,
              "figures": [rel(figs / "ch07_loss.png")],
              "meta": run_metadata(device, seed=args.seed, args=vars(args))}
    print("saved", rel(save_json(result, out_dir / "ch07_instruct.json")))


if __name__ == "__main__":
    main()
