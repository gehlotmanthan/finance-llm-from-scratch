"""Capstone step 1 — score every headline in the backtest window and fetch prices.

* Universe: the N tickers with the most headlines in the PRETRAINING window
  (chosen before the backtest periods start, so the choice uses no future data).
* Headlines dated after the pretraining window are scored with
    - the Ch 6 GPT-2 sentiment classifier selected on validation
    - the Ch 6 TF-IDF + logistic regression baseline
  score = P(positive) - P(negative).
* Daily adjusted prices for the universe and SPY are downloaded once (yfinance)
  and cached in data/prices/.

Output: data/processed/news_scores.parquet, results/capstone_scoring.json
"""

import joblib
import numpy as np
import pandas as pd
import torch

from _common import base_parser, news_and_periods, rel
from finllm.backtest.prices import download_prices
from finllm.backtest.signal import sentiment_score
from finllm.classifier import GPTClassifier, predict_proba
from finllm.config import TINY_CONFIG
from finllm.model import GPTModel
from finllm.paths import PRICES_DIR, PROCESSED_DIR, checkpoint_path, results_dirs
from finllm.tokenizer import get_gpt2_tokenizer
from finllm.utils import Timer, device_name, get_device, run_metadata, save_json, set_seed


def load_sentiment_model(device, smoke):
    path = checkpoint_path("sentiment-best.pth", smoke)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/ch06_sentiment.py first")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    clf = GPTClassifier(GPTModel(ck["cfg"]), len(ck["labels"]))
    clf.load_state_dict(ck["model_state_dict"])
    return clf.to(device).eval(), ck


def main():
    p = base_parser(__doc__)
    p.add_argument("--top-n", type=int, default=750, help="universe size")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--skip-prices", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, _ = results_dirs(args.smoke)
    tok = get_gpt2_tokenizer()

    news, periods = news_and_periods(args.smoke)
    pre = news[news["date"] <= periods["pretrain_end"]]
    universe = pre["ticker"].value_counts().head(args.top_n).index.tolist()
    bt = news[(news["date"] > periods["pretrain_end"]) & news["ticker"].isin(universe)].copy()
    print(f"universe {len(universe)} tickers | {len(bt):,} headlines from "
          f"{periods['pretrain_end'].date()} to {periods['end'].date()}")

    clf, ck = load_sentiment_model(device, args.smoke)
    texts = bt["headline"].tolist()
    with Timer() as t:
        probs = predict_proba(clf, texts, tok, device, max_length=ck["max_length"], batch_size=args.batch_size,
                              show_progress=not args.smoke)
    bt["gpt_score"] = sentiment_score(probs)
    bt["gpt_label"] = np.array(ck["labels"])[probs.argmax(1)]
    tfidf = joblib.load(checkpoint_path("tfidf_lr.joblib", args.smoke))
    bt["tfidf_score"] = sentiment_score(tfidf.predict_proba(texts))
    print(f"scored {len(bt):,} headlines in {t.seconds:.0f}s ({len(bt) / max(t.seconds, 1e-9):,.0f}/s) "
          f"with {ck['name']}")

    cols = ["date", "ticker", "headline", "gpt_score", "gpt_label", "tfidf_score"]
    if args.smoke:
        cols.append("true_label")
        dest = out_dir / "news_scores.parquet"
    else:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = PROCESSED_DIR / "news_scores.parquet"
    bt[cols].to_parquet(dest, index=False)

    price_info = None
    if not args.smoke and not args.skip_prices:
        price_info = download_prices(universe + ["SPY"], periods["pretrain_end"] - pd.Timedelta(days=10),
                                     periods["end"], PRICES_DIR)

    label_share = bt["gpt_label"].value_counts(normalize=True).to_dict()
    result = {"part": "capstone_scoring", "smoke": args.smoke, "device": device_name(device),
              "model": ck["name"], "universe_size": len(universe), "n_headlines": len(bt),
              "headlines_per_second": len(bt) / max(t.seconds, 1e-9), "gpt_label_share": label_share,
              "corr_gpt_vs_tfidf_score": float(np.corrcoef(bt["gpt_score"], bt["tfidf_score"])[0, 1]),
              "periods": {k: str(v.date()) for k, v in periods.items()},
              "prices": price_info, "scores_file": rel(dest),
              "meta": run_metadata(device, seed=args.seed, args=vars(args))}
    print("saved", rel(save_json(result, out_dir / "capstone_scoring.json")))


if __name__ == "__main__":
    main()
