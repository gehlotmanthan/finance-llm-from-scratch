# Methodology

This is the backtest protocol, written down in enough detail that someone can rerun it or pick holes in it.

## 1. Data and timeline

The cleaned news file has about 1.8M headlines, each with a ticker and a date. I split its calendar into three back-to-back windows. The split uses calendar time rather than row counts, so the heavier coverage in later years doesn't drag the boundaries forward.

| Window | Share of calendar | Used for |
|---|---|---|
| Pretraining | first 40% | Ch 5 pretraining from scratch and domain-adaptive pretraining (text only), plus choosing the universe. Never traded. |
| In-sample | next 30% | The first look at the backtest. Any design decision would have been made here. |
| Out-of-sample | last 30% | Run once with the settings already fixed in code. |

The actual dates are saved in `results/capstone_backtest.json` under `periods`. For the reported run they are Feb 2010 to Mar 2014, Mar 2014 to Apr 2017, and Apr 2017 to Jun 2020.

One bad row nearly broke this. A single headline dated 1969-12-31 (a "time zero" placeholder) stretched the calendar back by four decades, and the pretraining window ended up holding one headline. The cleaning step now drops dates before 2000, and `calendar_periods` refuses to run if any window holds less than 1% of the data.

## 2. Universe

The universe is the 750 tickers with the most headlines in the pretraining window, so picking it uses nothing from the trading windows. On any given day a stock can only be traded if its previous close was at least $5, which is known before the open.

## 3. Sentiment model

The signal comes from the Ch 6 GPT-2 classifier fine-tuned on Financial PhraseBank. Of the candidates (the book's recipe, full fine-tuning, a DAPT starting point, GPT-2 medium), it is the one with the best validation macro-F1. It never sees a price or a return.

Each headline gets the score `P(positive) − P(negative)`, which runs from −1 to 1. A stock's signal on trade day *t* is the average score of all its headlines that map to *t*.

The same pipeline runs with the TF-IDF + logistic regression baseline as well, to see whether the LLM adds anything over a bag of words.

## 4. Timing

This is the rule that matters most.

```
headline dated d  (time of day unknown, possibly after the close)
   └─► trade day t = first trading day strictly after d
         enter at OPEN(t), exit at OPEN(t+1)
```

Weekend and holiday news rolls into the next session, and nothing dated *d* is traded on *d*. Forward returns use opens only, and the tradability filter looks at the close of *t−1*.

The test `tests/test_backtest.py::test_backtest_finds_planted_signal_and_not_stale_signal` plants a return that the news predicts, and the backtest has to find it (t > 5). It then shifts the news one day later, so it only arrives after the move, and the backtest has to find nothing (|t| < 3).

## 5. Portfolio

Stocks scoring +0.25 or more go long and stocks at −0.25 or below go short. Each side is equal-weighted at 0.5 gross, so the book is dollar-neutral. It only trades on days with at least three names on each side, and positions are held for one day.

Turnover is `Σ|w_t − w_{t−1}|`, and the default cost is 10 bps per unit of turnover, with extra runs at 0, 2, 5 and 20 bps. Daily returns are clipped at ±40% so a bad tick can't dominate the result.

## 6. Statistics

| Question | What answers it |
|---|---|
| Is the mean daily return different from zero? | Newey-West t-stat (Bartlett kernel, automatic lag) |
| How uncertain is the Sharpe ratio? | 95% circular block-bootstrap interval, 10-day blocks |
| Would random picks on the same news days do as well? | Permutation test that shuffles scores across stocks within each day, 500 draws |
| Does the model rank stocks the right way round? | Daily Spearman rank IC: mean, t-stat, share of positive days |
| How fast does the information fade? | IC at the same-day close (not tradable), then from the next open over 1, 2, 3 and 5 days |
| Does anything survive trading costs? | Net Sharpe at each cost level |
| Is it just market exposure? | Beta of the gross book to SPY |

The IC profile is the most useful diagnostic. A high same-day IC with a low next-open IC means the model reads the news the way the market does, but the market has priced it in before the strategy can trade. That's what happened here, and it's worth knowing even though the strategy itself doesn't pay.

## 7. Known biases

Survivorship is the big one. Yahoo has no prices for delisted tickers, and only 412 of the 750 stocks had data. That usually flatters a strategy, so a fair reading of these results is, if anything, a little worse than the tables suggest.

Coverage is uneven, because partner-site headlines over-represent some publishers and some stocks.

Several model variants exist, which raises the risk of multiple testing. The capstone uses exactly one, picked on PhraseBank validation data, which limits how far the backtest itself could have been tuned.

The cost model is a flat bps charge with no market impact and no borrow fees. Shorting small caps would cost more than this assumes.

## 8. With more data or time

I'd start with timestamped news (RavenPack or Dow Jones, say) to test intraday reaction windows, since that's where the edge would have to be. CRSP prices would remove the survivorship bias, and sector or beta neutralisation would tighten the book. After that I'd combine the sentiment signal with momentum to see if it adds anything at the margin, and calibrate the classifier's probabilities so positions can scale with confidence.
