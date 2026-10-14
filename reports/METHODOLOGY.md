# Methodology: from headline sentiment to a tested trading signal

This note sets out the backtest protocol in enough detail to reproduce or attack it.

## 1. Data and timeline

The news file holds about 1.8M headlines with a ticker and a date. Its calendar span is split into three consecutive windows. The split is by **calendar time**, not by row count, so later years with heavier coverage do not pull the boundaries:

| Window | Share of calendar | Used for |
|---|---|---|
| Pretraining | first 40% | Ch 5 from-scratch pretraining and domain-adaptive pretraining (text only). Universe selection. Never traded. |
| In-sample | next 30% | First look at the backtest. Any design choice would be made here. |
| Out-of-sample | last 30% | Reported once, with the settings frozen in code. |

The exact dates are written to `results/capstone_backtest.json` (`periods`).

## 2. Universe

The universe is the 750 tickers with the most headlines in the pretraining window, so membership uses no information from the trading windows. On each day a stock is tradable only if its **previous** close was at least $5, a rule known before the open.

## 3. Sentiment model

The model is a GPT-2 classifier fine-tuned on Financial PhraseBank (Ch 6). Among the candidate models (book recipe, full fine-tune, DAPT start, GPT-2 medium) it is chosen by **validation** macro-F1. It never sees prices or returns.

Each headline gets a score `P(positive) − P(negative)` ∈ [−1, 1]. A stock's signal on trade day *t* is the mean score of all its headlines mapped to *t*.

The same pipeline runs with the TF-IDF + logistic-regression baseline. The comparison asks whether the LLM adds anything over a bag of words.

## 4. Timing (the most important rule)

```
headline dated d  (time of day unknown, possibly after the close)
   └─► trade day t = first trading day strictly after d
         enter at OPEN(t), exit at OPEN(t+1)
```

- News from weekends and holidays rolls into the next session.
- Nothing dated *d* is traded on *d*.
- Forward returns use opens only. The tradability filter uses the close of *t−1*.
- `tests/test_backtest.py::test_backtest_finds_planted_signal_and_not_stale_signal` plants a return that the news predicts, then shifts the news one day later so it is only knowable after the move. The backtest must find the first signal (t > 5) and nothing in the second (|t| < 3).

## 5. Portfolio

- Long names with score ≥ +0.25 and short names with score ≤ −0.25. Each side is equal-weighted with 0.5 gross, so the book is dollar-neutral.
- The book trades only on days with at least 3 names on each side.
- Positions are held for one day. Turnover is `Σ|w_t − w_{t−1}|`, and the cost is 10 bps per unit of turnover, with sensitivity runs at 0, 2, 5, 10 and 20 bps.
- Daily returns are clipped at ±40% to neutralise bad ticks.

## 6. Statistics reported

| Question | Statistic |
|---|---|
| Is the mean daily return different from zero? | Newey-West t-stat (Bartlett kernel, automatic lag) |
| How uncertain is the Sharpe ratio? | 95% circular block-bootstrap CI (10-day blocks) |
| Could random stock picking on the same news days do as well? | Permutation test: shuffle scores across stocks within each day (500 draws) |
| Does the model rank stocks correctly? | Daily Spearman rank IC: mean, t-stat, share of positive days |
| How fast does the information decay? | IC at the same-day close (not tradable), then next open +1, +2, +3 and +5 days |
| Does the edge survive trading? | Net Sharpe vs cost per side |
| Is it just market exposure? | Beta of the gross book to SPY |

How to read the IC profile: a large same-day IC with a small next-open IC means the model reads news the way the market does, and the market has already priced it by the next open. That is an informative result even when the strategy does not pay.

## 7. Known biases and threats

1. **Survivorship.** Yahoo has no prices for delisted tickers. The direction of the bias is usually favourable to the strategy, and coverage is reported.
2. **Coverage selection.** Partner-site headlines over-represent some publishers and stocks.
3. **Multiple testing.** Several model variants exist. The capstone uses one model, chosen on PhraseBank validation, which limits how much the backtest itself can be tuned.
4. **Costs.** A flat bps charge omits impact and borrow costs. Small-cap shorts would cost more.

## 8. What I would do with more data or time

- Timestamped news (e.g. RavenPack or Dow Jones) to test intraday reaction windows.
- CRSP prices to remove survivorship bias, and sector or beta neutralisation.
- Combining the sentiment signal with momentum, and testing for incremental alpha.
- Calibrating the classifier's probabilities and weighting positions by confidence.
