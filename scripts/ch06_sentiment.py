"""Chapter 6 — fine-tune GPT-2 as a financial-sentiment classifier.

Data: Financial PhraseBank (sentences from financial news labelled
negative / neutral / positive by 16 annotators), de-duplicated, stratified
70/10/20 train/val/test split that is fixed and shared with Ch 7.

Compared on the SAME test set, with 95% bootstrap confidence intervals:
  * majority class baseline
  * TF-IDF + logistic regression (C picked on validation)
  * GPT-2 small, last block trained (the book's recipe)
  * GPT-2 small, all layers trained
  * GPT-2 small after finance DAPT (Ch 5), all layers trained  [if checkpoint exists]
  * GPT-2 medium, last block trained

Model selection uses validation macro-F1 only; the test set is touched once
per model. The best model (by validation) is saved for the capstone backtest.

Output: results/ch06_sentiment.json, 3 figures, checkpoints/sentiment-best.pth, checkpoints/tfidf_lr.joblib
"""

import joblib
import numpy as np
import torch

from _common import base_parser, load_base_gpt2, plt, rel, savefig, SERIES, SEQ_BLUES, INK_2
from finllm import data_sources as ds
from finllm import synthetic
from finllm.baselines import fit_tfidf_lr, majority_predict
from finllm.classifier import (GPTClassifier, make_loaders, predict_logits_loader, set_trainable,
                               train_classifier)
from finllm.model import GPTModel
from finllm.paths import checkpoint_path, results_dirs
from finllm.stats import (accuracy, bootstrap_ci, classification_metrics, macro_f1, mcnemar_exact,
                          paired_bootstrap_diff)
from finllm.tokenizer import get_gpt2_tokenizer
from finllm.train import load_checkpoint
from finllm.utils import Timer, device_name, get_device, run_metadata, save_json, set_seed
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_VARIANTS = ["gpt2-small:last_block", "gpt2-small:all", "dapt:all", "gpt2-medium:last_block"]


def load_backbone(base: str, device, smoke: bool):
    if base == "dapt":
        path = checkpoint_path("gpt2-small-dapt.pth", smoke)
        if not path.exists():
            return None, None
        ck = load_checkpoint(path)
        cfg = dict(ck["cfg"], drop_rate=0.0)
        m = GPTModel(cfg)
        m.load_state_dict(ck["model_state_dict"])
        return m.to(device), cfg
    return load_base_gpt2(base, device, smoke, use_sdpa=True, drop_rate=0.0)


def summarise(y, pred, labels, seed):
    m = classification_metrics(y, pred, labels)
    m["accuracy_ci"] = bootstrap_ci(accuracy, y, pred, seed=seed)
    m["macro_f1_ci"] = bootstrap_ci(lambda a, b: macro_f1(a, b, len(labels)), y, pred, seed=seed)
    return m


def by_agreement(test_df, pred):
    out = {}
    for level, g in test_df.assign(pred=pred).groupby("agreement"):
        out[str(level)] = {"n": int(len(g)), "accuracy": float((g["pred"] == g["label_id"]).mean())}
    return out


def main():
    p = base_parser(__doc__)
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, figs = results_dirs(args.smoke)
    tok = get_gpt2_tokenizer()
    labels = ds.LABELS

    if args.smoke:
        splits = ds.split_phrasebank(synthetic.phrasebank_like(300), seed=args.seed)
        variants = ["gpt2-small:last_block", "dapt:all"]
        epochs = 1
    else:
        splits = ds.phrasebank_splits(seed=args.seed)
        variants, epochs = args.variants, args.epochs
    test_df = splits["test"]
    y_test = test_df["label_id"].to_numpy()
    data_info = {k: {"n": len(v), "class_counts": v["label"].value_counts().to_dict()} for k, v in splits.items()}
    print("split sizes:", {k: v["n"] for k, v in data_info.items()})

    models, preds = {}, {}
    # Baselines
    pred = majority_predict(splits["train"]["label_id"], len(test_df))
    models["Majority class"] = {"test": summarise(y_test, pred, labels, args.seed)}
    preds["Majority class"] = pred

    tfidf, info = fit_tfidf_lr(splits["train"]["text"].tolist(), splits["train"]["label_id"].to_numpy(),
                               splits["val"]["text"].tolist(), splits["val"]["label_id"].to_numpy(), seed=args.seed)
    pred = tfidf.predict(test_df["text"].tolist())
    models["TF-IDF + LogReg"] = {"test": summarise(y_test, pred, labels, args.seed), **info,
                                 "by_agreement": by_agreement(test_df, pred)}
    preds["TF-IDF + LogReg"] = pred
    joblib.dump(tfidf, checkpoint_path("tfidf_lr.joblib", args.smoke))
    print(f"TF-IDF + LogReg test macro-F1 {models['TF-IDF + LogReg']['test']['macro_f1']:.3f}")

    loaders, max_len = make_loaders(splits, tok, batch_size=args.batch_size, seed=args.seed)
    best_val, best_name = -1.0, None
    for v in variants:
        base, mode = v.split(":")
        set_seed(args.seed)
        gpt, cfg = load_backbone(base, device, args.smoke)
        if gpt is None:
            print(f"skip {v}: no DAPT checkpoint (run scripts/ch05_dapt.py first)")
            continue
        name = f"GPT-2 {base.replace('gpt2-', '')} ({mode.replace('_', ' ')})" if base != "dapt" \
            else f"GPT-2 small + DAPT ({mode.replace('_', ' ')})"
        print(f"\n== {name}")
        clf = GPTClassifier(gpt, len(labels)).to(device)
        n_train = set_trainable(clf, mode)
        lr = 5e-5 if mode != "all" else 2e-5
        opt = torch.optim.AdamW([q for q in clf.parameters() if q.requires_grad], lr=lr, weight_decay=0.1)
        with Timer() as t:
            hist = train_classifier(clf, loaders["train"], loaders["val"], opt, device, num_epochs=epochs)
        logits, _ = predict_logits_loader(clf, loaders["test"], device)
        pred = logits.argmax(1)
        preds[name] = pred
        models[name] = {"variant": v, "trainable_params": n_train, "lr": lr, "train_seconds": t.seconds,
                        "history": hist, "test": summarise(y_test, pred, labels, args.seed),
                        "by_agreement": by_agreement(test_df, pred),
                        "vs_tfidf_macro_f1": paired_bootstrap_diff(lambda a, b: macro_f1(a, b, 3), y_test, pred,
                                                                   preds["TF-IDF + LogReg"], seed=args.seed),
                        "vs_tfidf_mcnemar": mcnemar_exact(y_test, pred, preds["TF-IDF + LogReg"])}
        te = models[name]["test"]
        print(f"  test acc {te['accuracy']:.3f}  macro-F1 {te['macro_f1']:.3f} "
              f"[{te['macro_f1_ci'][0]:.3f}, {te['macro_f1_ci'][1]:.3f}]")
        if hist["best_val_macro_f1"] > best_val:
            best_val, best_name = hist["best_val_macro_f1"], name
            torch.save({"model_state_dict": clf.state_dict(), "cfg": cfg, "variant": v, "name": name,
                        "labels": labels, "max_length": max_len, "val_macro_f1": best_val},
                       checkpoint_path("sentiment-best.pth", args.smoke))
        del clf, gpt, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nbest by validation macro-F1: {best_name}")

    # Figure 1: model comparison
    names = list(models)
    f1 = [models[n]["test"]["macro_f1"] for n in names]
    lo = [f - models[n]["test"]["macro_f1_ci"][0] for n, f in zip(names, f1)]
    hi = [models[n]["test"]["macro_f1_ci"][1] - f for n, f in zip(names, f1)]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(names) + 1.2))
    colors = [INK_2 if n in ("Majority class", "TF-IDF + LogReg") else SERIES[0] for n in names]
    ax.barh(names, f1, xerr=[lo, hi], color=colors, height=0.55, error_kw={"ecolor": "#0b0b0b", "lw": 1})
    for i, f in enumerate(f1):
        ax.text(f + hi[i] + 0.01, i, f"{f:.3f}", va="center", fontsize=8, color=INK_2)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.08), xlabel="test macro-F1 (95% bootstrap CI)",
           title="Financial sentiment: fine-tuned GPT-2 vs baselines")
    ax.grid(axis="y", visible=False)
    savefig(fig, figs, "ch06_model_comparison.png")

    # Figure 2: confusion matrix of the selected model
    cm = np.array(models[best_name]["test"]["confusion_matrix"])
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cmn, cmap=LinearSegmentedColormap.from_list("b", ["#fcfcfb"] + SEQ_BLUES), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.6 else "#0b0b0b")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set(xlabel="predicted", ylabel="true", title=best_name)
    ax.grid(False)
    savefig(fig, figs, "ch06_confusion.png")

    # Figure 3: accuracy by annotator agreement
    levels = sorted(models["TF-IDF + LogReg"]["by_agreement"], key=int)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs = np.arange(len(levels))
    for i, n in enumerate(["TF-IDF + LogReg", best_name]):
        ax.bar(xs + (i - 0.5) * 0.38, [models[n]["by_agreement"][l]["accuracy"] for l in levels], 0.36,
               color=[INK_2, SERIES[0]][i], label=n)
    ax.set_xticks(xs, [f"{l}%\n(n={models[best_name]['by_agreement'][l]['n']})" for l in levels])
    ax.set(ylim=(0, 1.05), xlabel="share of annotators who agreed on the label", ylabel="test accuracy",
           title="Test accuracy by annotator agreement")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    savefig(fig, figs, "ch06_by_agreement.png")

    result = {"chapter": 6, "smoke": args.smoke, "device": device_name(device), "labels": labels,
              "data": data_info, "max_length_tokens": max_len, "models": models,
              "selected_by_validation": best_name,
              "checkpoints": [rel(checkpoint_path("sentiment-best.pth", args.smoke)),
                              rel(checkpoint_path("tfidf_lr.joblib", args.smoke))],
              "figures": [rel(figs / f) for f in ("ch06_model_comparison.png", "ch06_confusion.png",
                                                  "ch06_by_agreement.png")],
              "meta": run_metadata(device, seed=args.seed, args=vars(args))}
    print("saved", rel(save_json(result, out_dir / "ch06_sentiment.json")))


if __name__ == "__main__":
    main()
