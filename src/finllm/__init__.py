"""finllm: a small GPT-style language model applied to financial text.

    tokenizer.py, data.py        text -> BPE tokens -> (input, target) training windows
    attention.py                 causal multi-head self-attention (explicit and fused)
    model.py, generate.py        GPT-2 architecture, parameter accounting, decoding
    train.py, gpt2_weights.py    language-model training loop, loading OpenAI's GPT-2 weights
    classifier.py, baselines.py  financial-sentiment classifier and the TF-IDF baseline
    instruct.py, lora.py         instruction tuning, low-rank adaptation for small GPUs
    backtest/                    headline sentiment -> daily long/short signal -> evaluation
    stats.py                     bootstrap, McNemar, Newey-West, Sharpe intervals
"""

__version__ = "0.1.0"
