"""Capstone step 2 — does LLM headline sentiment predict next-day stock returns?

Design (all fixed before looking at the out-of-sample period):
  * signal: mean headline score per stock, news from day d traded at the
    OPEN of the next trading day, held to the next open (1 day)
  * book: dollar-neutral long/short, threshold +-0.25, >= 3 names per side,
    stocks with prior close >= $5, 10 bps per side transaction cost
  * periods: in-sample = next 30% of the news calendar after the pretraining
    window; out-of-sample = final 30%, reported once
  * comparisons: GPT-2 classifier vs TF-IDF baseline on the same rules
  * evidence: Newey-West t-stats, block-bootstrap Sharpe CI, within-day
    permutation test, information coefficient by horizon (incl. the same-day
    price reaction, which shows whether the model reads news the way the market
    does even when that is no longer tradable), cost sensitivity, beta to SPY

Output: results/capstone_backtest.json + 4 figures.
"""

import numpy as np
import pandas as pd

from _common import base_parser, news_and_periods, plt, rel, savefig, SERIES, INK_2, INK_3
from finllm import synthetic
from finllm.backtest.portfolio import (daily_ic, forward_open_returns, ic_summary, long_short_weights,
                                       permutation_test, run_book, same_day_close_returns, tradable_mask)
from finllm.backtest.prices import load_price_panel
from finllm.backtest.signal import build_signal, to_wide
from finllm.paths import PRICES_DIR, PROCESSED_DIR, results_dirs
from finllm.stats import block_bootstrap_sharpe_ci, perf_summary
from finllm.utils import run_metadata, save_json, set_seed

COSTS = [0, 2, 5, 10, 20]
HORIZONS = [1, 2, 3, 5]


def period_slice(df, start, end):
    return df[(df.index > start) & (df.index <= end)]


def main():
    p = base_parser(__doc__)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--min-names", type=int, default=3)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--prices-dir", default=str(PRICES_DIR))
    args = p.parse_args()
    set_seed(args.seed)
    out_dir, figs = results_dirs(args.smoke)

    news, periods = news_and_periods(args.smoke)
    if args.smoke:
        scores = pd.read_parquet(out_dir / "news_scores.parquet")
        scores["oracle_score"] = scores["true_label"] - 1.0      # planted signal: the backtest must find it
        opens, closes = synthetic.prices_like(news)
        spy = None
        models = {"GPT-2 classifier": "gpt_score", "TF-IDF + LogReg": "tfidf_score", "Oracle (smoke check)": "oracle_score"}
    else:
        scores = pd.read_parquet(PROCESSED_DIR / "news_scores.parquet")
        panel = load_price_panel(sorted(scores["ticker"].unique()) + ["SPY"], args.prices_dir)
        opens, closes = panel["open"], panel["close"]
        spy = opens.pop("SPY") if "SPY" in opens else None
        closes = closes.drop(columns=["SPY"], errors="ignore")
        models = {"GPT-2 classifier": "gpt_score", "TF-IDF + LogReg": "tfidf_score"}

    calendar = opens.index
    covered = scores["ticker"].isin(opens.columns)
    coverage = {"headlines_with_prices": float(covered.mean()),
                "tickers_with_prices": int(scores.loc[covered, "ticker"].nunique()),
                "tickers_total": int(scores["ticker"].nunique())}
    scores = scores[covered]
    print(f"price coverage: {coverage['headlines_with_prices']:.1%} of headlines, "
          f"{coverage['tickers_with_prices']}/{coverage['tickers_total']} tickers")

    tradable = tradable_mask(closes, args.min_price)
    fwd = {h: forward_open_returns(opens, h) for h in HORIZONS}
    same_day = same_day_close_returns(closes)
    spy_fwd = (spy.shift(-1) / spy - 1) if spy is not None else None
    windows = {"in_sample": (periods["pretrain_end"], periods["is_end"]),
               "out_of_sample": (periods["is_end"], periods["end"])}

    res = {}
    books = {}
    for mname, col in models.items():
        sig = build_signal(scores, calendar, col)
        score_w = to_wide(sig).reindex(columns=opens.columns)
        # same-day diagnostic uses the NEWS date, only for news on trading days
        day0 = scores[scores["date"].isin(calendar)].groupby(["date", "ticker"])[col].mean().unstack()
        w = long_short_weights(score_w, fwd[1], tradable, args.threshold, args.min_names)
        book = run_book(w, fwd[1], args.cost_bps)
        books[mname] = book
        res[mname] = {}
        for pname, (a, b) in windows.items():
            bk = period_slice(book, a, b)
            active = bk[(bk["n_long"] > 0)]
            net, gross = bk["net"].to_numpy(), bk["gross"].to_numpy()
            r = {"net": perf_summary(net), "gross": perf_summary(gross),
                 "net_sharpe_ci": block_bootstrap_sharpe_ci(net, seed=args.seed),
                 "active_days": int(len(active)), "days": int(len(bk)),
                 "avg_names_long": float(active["n_long"].mean()) if len(active) else 0.0,
                 "avg_names_short": float(active["n_short"].mean()) if len(active) else 0.0,
                 "avg_daily_turnover": float(bk["turnover"].mean()),
                 "long_leg_minus_short_leg_bps": float((active["long_leg"] - active["short_leg"]).mean() * 1e4) if len(active) else float("nan"),
                 "net_by_cost_bps": {}}
            for c in COSTS:
                r["net_by_cost_bps"][str(c)] = perf_summary(period_slice(run_book(w, fwd[1], c), a, b)["net"].to_numpy())["sharpe"]
            ic = {}
            for h in HORIZONS:
                ic[f"open_t_to_open_t+{h}"] = ic_summary(daily_ic(period_slice(score_w, a, b), fwd[h], tradable))
            ic["same_day_close_to_close (not tradable)"] = ic_summary(daily_ic(period_slice(day0, a, b), same_day))
            r["ic"] = ic
            if spy_fwd is not None:
                j = pd.concat([bk["gross"], spy_fwd.rename("spy")], axis=1, join="inner").dropna()
                j = j[j["gross"] != 0]
                r["beta_to_spy"] = float(np.polyfit(j["spy"], j["gross"], 1)[0]) if len(j) > 20 else float("nan")
            r["permutation"] = permutation_test(period_slice(score_w, a, b), fwd[1], tradable, args.threshold,
                                                args.min_names, n_perm=50 if args.smoke else args.n_perm, seed=args.seed)
            res[mname][pname] = r
            print(f"{mname:22s} {pname:14s} net Sharpe {r['net']['sharpe']:6.2f}  gross {r['gross']['sharpe']:6.2f}  "
                  f"IC(1d) {ic['open_t_to_open_t+1']['mean_ic']:+.4f}  same-day IC "
                  f"{ic['same_day_close_to_close (not tradable)']['mean_ic']:+.4f}  "
                  f"perm p {r['permutation']['p_value_one_sided']:.3f}")

    # ------------------------------------------------------------ figures
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (mname, bk) in enumerate(books.items()):
        bk = period_slice(bk, periods["pretrain_end"], periods["end"])
        ax.plot(bk.index, (1 + bk["net"]).cumprod(), color=SERIES[i], lw=1.6, label=mname)
    ax.axvline(periods["is_end"], color=INK_3, lw=1, ls="--")
    ax.text(periods["is_end"], ax.get_ylim()[1], " out-of-sample →", va="top", fontsize=8, color=INK_2)
    ax.set(ylabel=f"growth of 1 (net of {args.cost_bps:g} bps)", title="Sentiment long/short, next-open entry")
    ax.legend(loc="upper left")
    savefig(fig, figs, "capstone_equity.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    hz = ["same-day\n(close d)"] + [f"+{h}d\n(open)" for h in HORIZONS]
    for ax, pname in zip(axes, windows):
        width = 0.8 / len(models)
        for i, mname in enumerate(models):
            ic = res[mname][pname]["ic"]
            vals = [ic["same_day_close_to_close (not tradable)"]["mean_ic"]] + \
                   [ic[f"open_t_to_open_t+{h}"]["mean_ic"] for h in HORIZONS]
            ax.bar(np.arange(len(hz)) + i * width, vals, width * 0.92, color=SERIES[i], label=mname)
        ax.axhline(0, color=INK_3, lw=0.8)
        ax.set_xticks(np.arange(len(hz)) + width * (len(models) - 1) / 2, hz, fontsize=8)
        ax.set_title(pname.replace("_", "-"))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean daily rank IC")
    axes[0].legend(fontsize=8)
    savefig(fig, figs, "capstone_ic_decay.png")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, mname in enumerate(models):
        d = res[mname]["out_of_sample"]["net_by_cost_bps"]
        ax.plot([int(c) for c in d], list(d.values()), marker="o", ms=5, color=SERIES[i], label=mname)
    ax.axhline(0, color=INK_3, lw=0.8)
    ax.set(xlabel="transaction cost (bps per side)", ylabel="out-of-sample Sharpe",
           title="How much of the edge survives costs?")
    ax.legend()
    savefig(fig, figs, "capstone_costs.png")

    first = list(models)[0]
    perm = res[first]["out_of_sample"]["permutation"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(np.array(perm["null_sample"]) * 1e4, bins=40, color=SERIES[0], alpha=0.85)
    ax.axvline(perm["actual_mean_daily_gross"] * 1e4, color=SERIES[1], lw=2)
    ax.text(perm["actual_mean_daily_gross"] * 1e4, ax.get_ylim()[1] * 0.95,
            f" actual\n p = {perm['p_value_one_sided']:.3f}", va="top", fontsize=8, color=INK_2)
    ax.set(xlabel="mean daily gross return (bps) with scores shuffled within each day",
           ylabel="count", title=f"{first}: out-of-sample permutation test")
    savefig(fig, figs, "capstone_permutation.png")

    result = {"part": "capstone_backtest", "smoke": args.smoke,
              "settings": {"threshold": args.threshold, "min_names": args.min_names, "cost_bps": args.cost_bps,
                           "min_price": args.min_price, "entry": "open of first trading day after news date",
                           "holding": "1 day open-to-open", "return_clip": 0.4},
              "periods": {k: str(v.date()) for k, v in periods.items()}, "coverage": coverage,
              "results": res,
              "figures": [rel(figs / f) for f in ("capstone_equity.png", "capstone_ic_decay.png",
                                                  "capstone_costs.png", "capstone_permutation.png")],
              "caveats": ["Prices from Yahoo Finance: delisted companies are missing (survivorship bias).",
                          "Headline timestamps are dates only; entry is delayed to the next open to stay safe.",
                          "Partner-site headlines are not a complete news feed; coverage varies by stock and year.",
                          "Costs are a flat bps charge; no market impact or borrow fees for shorts."],
              "meta": run_metadata(seed=args.seed, args=vars(args))}
    print("saved", rel(save_json(result, out_dir / "capstone_backtest.json")))


if __name__ == "__main__":
    main()
