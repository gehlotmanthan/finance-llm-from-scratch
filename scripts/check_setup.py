"""Step 0 — check the environment and every real data source before the long runs.

Downloads (once, cached in data/): Financial PhraseBank, the news headlines,
finance-alpaca, GPT-2 small weights; fetches one month of SPY from Yahoo.
Each check prints OK or FAIL with the reason, and the calendar windows the
backtest will use are printed so you can sanity-check them.

Output: results/setup_check.json
"""

import traceback

import torch

from _common import base_parser, rel
from finllm import data_sources as ds
from finllm.paths import RESULTS_DIR
from finllm.tokenizer import get_gpt2_tokenizer
from finllm.utils import device_name, get_device, run_metadata, save_json


def main():
    args = base_parser(__doc__).parse_args()
    device = get_device(args.device)
    report = {}

    def check(name, fn):
        try:
            info = fn()
            report[name] = {"ok": True, **(info or {})}
            print(f"OK   {name}: {info}")
        except Exception as e:  # report and carry on so every problem shows up in one run
            report[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)

    check("torch", lambda: {"version": torch.__version__, "device": device_name(device),
                            "bf16": bool(device.type == "cuda" and torch.cuda.is_bf16_supported())})
    check("tiktoken", lambda: {"hello_ids": get_gpt2_tokenizer().encode("Hello, world")})

    def phrasebank():
        df = ds.load_phrasebank()
        return {"sentences": len(df), "labels": df["label"].value_counts().to_dict(),
                "conflicting_dropped": df.attrs.get("n_conflicting_dropped"),
                "by_agreement": df["agreement"].value_counts().sort_index().to_dict()}
    check("phrasebank", phrasebank)

    def news():
        df = ds.load_news_cached()
        per = ds.calendar_periods(df["date"])
        return {"headlines": len(df), "tickers": int(df["ticker"].nunique()),
                "periods": {k: str(v.date()) for k, v in per.items()},
                "headlines_per_window": {
                    "pretrain": int((df["date"] <= per["pretrain_end"]).sum()),
                    "in_sample": int(((df["date"] > per["pretrain_end"]) & (df["date"] <= per["is_end"])).sum()),
                    "out_of_sample": int((df["date"] > per["is_end"]).sum())}}
    check("news", news)
    check("finance_alpaca", lambda: {"examples": len(ds.load_finance_alpaca())})

    def gpt2():
        from finllm.gpt2_weights import load_pretrained_gpt2
        from finllm.train import corpus_loss
        model, _ = load_pretrained_gpt2("gpt2-small (124M)", device=device)
        r = corpus_loss(model, ds.load_the_verdict(), get_gpt2_tokenizer(), device, max_length=256)
        if r["perplexity"] > 200:
            raise ValueError(f"perplexity {r['perplexity']:.0f} on The Verdict - weights not loaded correctly")
        return {"verdict_perplexity": round(r["perplexity"], 1)}
    check("gpt2_weights", gpt2)

    def yahoo():
        import yfinance as yf
        df = yf.download("SPY", period="1mo", auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("empty response from Yahoo Finance")
        return {"rows": len(df)}
    check("yahoo_prices", yahoo)

    ok = all(v["ok"] for v in report.values())
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - see above")
    save_json({"all_ok": ok, "checks": report, "meta": run_metadata(device)}, RESULTS_DIR / "setup_check.json")
    print("saved", rel(RESULTS_DIR / "setup_check.json"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
