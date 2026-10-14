# Finance LLM from Scratch

A GPT-2-style language model written from first principles in PyTorch, adapted to financial text, then tested the way a systematic researcher would test any signal: on data it has never seen, against simple baselines, net of costs, with significance tests.

Built chapter by chapter alongside Sebastian Raschka's *Build a Large Language Model (From Scratch)*. Every chapter is retargeted at a finance question, and the project ends with a capstone: **does LLM-read headline sentiment predict next-day stock returns once you trade it honestly?**

## Results

<!-- RESULTS:START -->
_No results yet. Run `.\run_all.ps1` (Windows) to produce them; this block fills in automatically._

Full tables and figures: [reports/RESULTS.md](reports/RESULTS.md).
<!-- RESULTS:END -->

Every number above is written by `scripts/make_report.py` from the JSON files in `results/`. Nothing is typed in by hand.

## Chapter map

| Ch | Book topic | Finance question asked here | Script |
|---|---|---|---|
| 2 | Tokenisation, BPE, data loaders | How efficiently does GPT-2's tokenizer encode financial news? How does it split tickers, `EBITDA`, `$1.2bn`? Why does a word-level vocabulary fail? | `ch02_tokenization.py` |
| 3 | Causal multi-head attention | Is the causal mask leak-free? It is the same no-look-ahead discipline a backtest needs. How do compute and memory scale with context, and what does the fused kernel save? | `ch03_attention.py` |
| 4 | GPT-2 architecture | What does each model size cost in memory, and what batch sizes fit on an 8 GB GPU? These are measured on the GPU, not estimated. | `ch04_model.py` |
| 5 | Pretraining, loading OpenAI weights | Train a 30M-parameter GPT on headlines from scratch. Measure GPT-2's "domain gap" on finance text. Run domain-adaptive pretraining and track the gain vs the forgetting. | `ch05_*.py` |
| 6 | Fine-tuning for classification | 3-class financial sentiment on Financial PhraseBank. Does GPT-2 beat TF-IDF + logistic regression with statistical significance? Does domain pretraining help? | `ch06_sentiment.py` |
| 7 | Instruction tuning (+ App. E LoRA) | A small finance assistant on an 8 GB GPU. Scored objectively on the *same* sentiment test set as Ch 6. | `ch07_instruct.py` |
| ★ | Capstone | Sentiment → daily dollar-neutral long/short. In-sample/out-of-sample, costs, IC decay, permutation test. | `capstone_*.py` |

## Research design: what keeps the backtest honest

- **No look-ahead in time.** Headlines carry a date but no time, so a story dated day *d* is traded at the **open of the next trading day**. A unit test plants a signal that is only knowable after the move it predicts, and checks that the backtest finds nothing.
- **No look-ahead in the text.** Domain pretraining (Ch 5) only reads headlines from the first 40% of the news calendar, a window that is never traded. The sentiment model is trained on PhraseBank labels, never on returns.
- **Out-of-sample once.** The thresholds, costs, holding period and universe rule are fixed in code before the last 30% of the calendar is looked at. That window is reported as is.
- **Evidence, not one Sharpe ratio.** The capstone reports Newey-West t-stats, a block-bootstrap Sharpe CI, a within-day permutation test, IC by horizon (including the not-tradable same-day reaction) and a cost-sensitivity curve. Classification results report bootstrap CIs, a paired bootstrap against the baseline and a McNemar test.
- **Model selection on validation only.** The Ch 6 model used in the capstone is picked on validation macro-F1. The test set is touched once per model.

Details: [reports/METHODOLOGY.md](reports/METHODOLOGY.md).

## Engineering notes

- **Model code is my own implementation** (`src/finllm/`), with no `transformers` dependency. OpenAI's released GPT-2 weights are mapped into it (`gpt2_weights.py`), and a test round-trips the mapping.
- **48 unit tests**, all running on CPU in about 15 s. They cover the causal mask, explicit vs fused attention equivalence, exact GPT-2 parameter counts, the weight mapping, LoRA starting as an identity, prompt masking, dynamic padding matching unpadded inference, the look-ahead check, and the portfolio arithmetic.
- **`--smoke` mode** runs the whole pipeline end to end on synthetic data in about 2 minutes on a CPU, with no downloads. The synthetic news contains a planted "oracle" signal that the backtest must recover.
- **Fits an 8 GB GPU:** fused scaled-dot-product attention, bf16 autocast, gradient accumulation, and LoRA (rank 16, about 2% of GPT-2 medium's weights trainable) for instruction tuning.
- **Reproducible:** fixed seeds, and every result file records the git commit, device, library versions and CLI arguments.

## Quick start

```powershell
# Windows. Uses ..\.venv if present; install a CUDA build of PyTorch first.
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Smoke   # 2-3 min pipeline check
powershell -ExecutionPolicy Bypass -File .\run_all.ps1          # full run, ~1.5-2.5 h on an 8 GB GPU
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -From ch06   # resume from a step
```

```bash
# Linux / macOS
pip install -r requirements.txt
./run_all.sh --smoke && ./run_all.sh
```

Single chapters run on their own: `python scripts/ch06_sentiment.py --help`.

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
scripts/               one script per chapter step + check_setup.py + make_report.py
tests/                 pytest suite
results/               JSON results + figures (committed)
reports/               RESULTS.md (generated), METHODOLOGY.md
data/ checkpoints/     downloads and weights (git-ignored)
```

## Data and licences

| Data | Source | Licence / note |
|---|---|---|
| Financial PhraseBank v1.0 (Malo et al., 2014) | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | CC BY-NC-SA 3.0, non-commercial |
| Financial news headlines with ticker and date | [ashraq/financial-news](https://huggingface.co/datasets/ashraq/financial-news), a mirror of the Kaggle "Massive Stock News Analysis DB" | Licence not stated on the card; used for research only and not redistributed |
| finance-alpaca | [gbharti/finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) | MIT |
| GPT-2 weights | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | Modified MIT (OpenAI) |
| Daily prices | Yahoo Finance via `yfinance` | Personal and research use |
| *The Verdict* (Edith Wharton, 1908) | book repository | Public domain |

The scripts download all data. No dataset is committed to this repository.

## Limitations

- **Survivorship bias:** Yahoo serves only tickers that still exist, so delisted names are missing. Coverage is reported. With CRSP access, pass `--prices-dir`.
- **Date-only timestamps** force a next-open entry. Any intraday edge is invisible here by construction.
- **Partner-site headlines** (Zacks, GuruFocus, Seeking Alpha, ...) are not a complete newswire, and coverage varies across stocks and years.
- **Costs** are a flat charge per side, with no market impact and no short-borrow fees.
- **GPT-2 is a 2019 model.** The point is the pipeline and the evaluation, not state-of-the-art accuracy.

## Acknowledgements

The model, training and fine-tuning code follow the implementations in Sebastian Raschka's *Build a Large Language Model (From Scratch)* (Manning, 2024) and its companion repository [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0). The financial retargeting, the experiments, the statistics and the backtest are this project's own.
