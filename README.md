# Finance LLM from Scratch

This is a GPT-2-style language model that I wrote in PyTorch and aimed at financial text. I then tested it the way I would test any trading signal: on data it had never seen, against a plain baseline, and after costs.

I built it while working through Sebastian Raschka's *Build a Large Language Model (From Scratch)*. Each chapter got a finance question attached. The last one is the question I care about most: once you trade it honestly, does headline sentiment read by an LLM predict next-day stock returns?

Everything here ran on an RTX 3060 laptop GPU with 6 GB of memory. That limit shaped a lot of the choices below.

## Results

<!-- RESULTS:START -->
| Experiment | Result |
|---|---|
| Ch 6, sentiment on held-out PhraseBank | macro-F1 0.847 [0.820, 0.873] for GPT-2 small + DAPT (all), against 0.748 for TF-IDF + logistic regression (McNemar p < 0.0001) |
| Ch 5, domain pretraining | perplexity on unseen headlines -75%, on general English +311% |
| Ch 7, instruction tuning | gpt2-medium (355M) with LoRA r=16 answers sentiment questions at 84.0% accuracy (untuned: 0.1%) |
| Capstone, same-day link to returns | rank IC 0.035 (t = 9.5, out-of-sample) |
| Capstone, next-open long/short | next-day IC 0.008, gross Sharpe 0.62, net Sharpe -5.62 [-7.10, -4.26] at 10 bps; break-even about 1.0 bp per side; permutation p = 0.10 |

Full tables and figures: [reports/RESULTS.md](reports/RESULTS.md).
<!-- RESULTS:END -->

`scripts/make_report.py` writes the table above from the JSON files in `results/`, so none of the numbers are typed in by hand.

### Reading the results

The sentiment classifier is the clear success. Fine-tuned GPT-2 beats TF-IDF plus logistic regression by about 0.10 macro-F1, and both McNemar's test and a paired bootstrap rule out luck. Most of its mistakes are on sentences the human annotators also disagreed about. Accuracy is 96% where every annotator agreed and 68% where only half of them did.

Domain pretraining helped less than I expected. Training GPT-2 further on 2010–14 headlines cut perplexity on unseen 2017–20 headlines by three quarters. Perplexity on ordinary English and on PhraseBank's news sentences went up four- to five-fold, though. The model picked up headline style, lost some general language, and the sentiment score hardly moved (0.845 to 0.847).

The trading test is where the data ran out. Sentiment tracks same-day returns, with a rank IC of about 0.035 and t-stats of 6.9 and 9.5 in the two periods, so the model reads news roughly the way the market does. The problem is timing. The headlines have dates but no times, so the earliest honest entry is the next morning's open, and by then most of the move has already happened. Next-day IC is 0.002 in-sample and 0.008 out-of-sample (t = 2.0), and the permutation test gives p = 0.10. A one-day hold also means turning the book over about 1.8 times a day. Out of sample that breaks even at roughly 1 bp per side, and at 10 bps it loses money in both periods.

I read that as a data limit more than a model limit. Free Yahoo data had prices for only 412 of the 750 stocks. Date-only headlines also hide the intraday window where news is actually priced. Next I'd try timestamped news, CRSP prices and slower multi-day signals, each tested on a fresh holdout, since I've now looked at this one.

## Chapters

| Ch | Book topic | Finance question | Script |
|---|---|---|---|
| 2 | Tokenisation, BPE, data loaders | How efficiently does GPT-2's tokenizer encode financial news? How does it split tickers, `EBITDA` or `$1.2bn`? Why does a word-level vocabulary break? | `ch02_tokenization.py` |
| 3 | Causal multi-head attention | Does the causal mask leak anything? This is the same no-look-ahead rule a backtest needs. How do time and memory grow with context length, and what does the fused kernel save? | `ch03_attention.py` |
| 4 | GPT-2 architecture | What does each model size cost in memory, and what actually fits on a 6 GB laptop GPU? Measured, not estimated. | `ch04_model.py` |
| 5 | Pretraining, loading OpenAI weights | A 49M-parameter GPT trained from scratch on headlines, GPT-2's perplexity on finance text vs ordinary English, and domain-adaptive pretraining. | `ch05_*.py` |
| 6 | Fine-tuning for classification | Three-class financial sentiment on Financial PhraseBank. Does GPT-2 beat TF-IDF + logistic regression by more than noise? Does domain pretraining help? | `ch06_sentiment.py` |
| 7 | Instruction tuning, plus LoRA (App. E) | A small finance assistant on a 6 GB laptop GPU, scored on the same sentiment test set as Ch 6. | `ch07_instruct.py` |
| ★ | Capstone | Sentiment turned into a daily dollar-neutral long/short book, with an in-sample/out-of-sample split, costs, IC decay and a permutation test. | `capstone_*.py` |

## How the backtest avoids fooling itself

A story dated day *d* is traded at the open of the next trading day, because a date-only headline could have come out after the close. A unit test plants a signal that only becomes known after the move it predicts, then checks that the backtest finds nothing.

The text side follows the same rule. Domain pretraining only reads headlines from the first 40% of the news calendar, a stretch that is never traded. The sentiment model learns from PhraseBank labels and never sees a return.

Thresholds, costs, holding period and the universe rule were fixed in code before I opened the last 30% of the calendar. That window is reported as it came out. The Ch 6 model used in the backtest was chosen on validation macro-F1, and each model saw the test set once.

A single Sharpe ratio isn't enough evidence, so the backtest also reports Newey-West t-stats, a block-bootstrap confidence interval for the Sharpe, a within-day permutation test, and IC at several horizons. That includes the same-day reaction, which can't be traded but shows whether the model reads the news correctly. There is also a cost curve. The classification results come with bootstrap intervals, a paired bootstrap against the baseline and McNemar's test. [reports/METHODOLOGY.md](reports/METHODOLOGY.md) has the details.

## Engineering notes

The model code in `src/finllm/` has no `transformers` dependency. `gpt2_weights.py` maps OpenAI's released GPT-2 weights into it, and a test checks the mapping round-trips. The 49 unit tests take about 15 seconds on a CPU. They cover the causal mask, explicit vs fused attention, the exact GPT-2 parameter counts, LoRA starting out as an identity, prompt masking, dynamic padding, the look-ahead check and the portfolio arithmetic.

`--smoke` runs the full pipeline on synthetic data in about two minutes, with no downloads. The fake news contains a planted signal, and the backtest has to recover it.

Fitting into 6 GB took fused attention, bf16, gradient accumulation and LoRA (rank 16, 1.7% of GPT-2 medium's weights trainable). Full fine-tuning of GPT-2 medium doesn't fit at any batch size. On Windows an oversized run doesn't fail. It spills into system RAM and crawls, which is why `ch04_model.py` measures memory instead of trusting the arithmetic.

Seeds are fixed. Each result file records the git commit, device, library versions and command-line arguments.

## Running it

```powershell
# Windows. Uses ..\.venv if present; install a CUDA build of PyTorch first.
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Smoke        # 2-3 min pipeline check
powershell -ExecutionPolicy Bypass -File .\run_all.ps1               # full run, ~1.5 h on an RTX 3060 Laptop (6 GB)
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -From ch06    # resume from a step
```

```bash
# Linux / macOS
pip install -r requirements.txt
./run_all.sh --smoke && ./run_all.sh
```

Each chapter also runs on its own, e.g. `python scripts/ch06_sentiment.py --help`.

## Repository layout

```
src/finllm/            the library
  tokenizer.py data.py          Ch 2
  attention.py                  Ch 3
  model.py generate.py          Ch 4
  train.py gpt2_weights.py      Ch 5
  classifier.py baselines.py    Ch 6
  instruct.py lora.py           Ch 7 / App. E
  backtest/                     capstone: prices, signal alignment, portfolio, diagnostics
  stats.py                      bootstrap, McNemar, Newey-West, Sharpe CI
  data_sources.py synthetic.py  real datasets / synthetic stand-ins for tests
scripts/               one script per chapter step, plus check_setup.py and make_report.py
tests/                 pytest suite
results/               JSON results and figures
reports/               RESULTS.md (generated), METHODOLOGY.md
data/ checkpoints/     downloads and weights (not in git)
```

## Data and licences

| Data | Source | Licence / note |
|---|---|---|
| Financial PhraseBank v1.0 (Malo et al., 2014) | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | CC BY-NC-SA 3.0, non-commercial |
| Financial news headlines with ticker and date | [ashraq/financial-news](https://huggingface.co/datasets/ashraq/financial-news), a mirror of the Kaggle "Massive Stock News Analysis DB" | No licence stated on the card; used for research only and not redistributed |
| finance-alpaca | [gbharti/finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) | MIT |
| GPT-2 weights | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | Modified MIT (OpenAI) |
| Daily prices | Yahoo Finance via `yfinance` | Personal and research use |
| *The Verdict* (Edith Wharton, 1908) | book repository | Public domain |

The scripts download everything, and no dataset is committed here.

## Limitations

- Yahoo only serves tickers that still exist, so delisted names are missing and the backtest has survivorship bias. Coverage is reported. If you have CRSP access, point `--prices-dir` at it.
- With date-only timestamps the earliest fair entry is the next open, so any intraday edge is invisible here by construction.
- The headlines come from partner sites (Zacks, GuruFocus, Seeking Alpha and others), not a full newswire, and coverage varies by stock and year.
- Costs are a flat charge per side. Market impact and short-borrow fees are not modelled.
- GPT-2 dates from 2019. The point of the project is the pipeline and the evaluation, not state-of-the-art accuracy.

## Acknowledgements

The model, training and fine-tuning code follow Sebastian Raschka's *Build a Large Language Model (From Scratch)* (Manning, 2024) and its companion repository [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0). The finance questions, experiments, statistics and backtest are my own.
