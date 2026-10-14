"""finllm — a GPT-style language model built from scratch, applied to financial text.

Chapter map (Raschka, *Build a Large Language Model (From Scratch)*):

    Ch 2  tokenizer.py, data.py      text -> tokens -> (input, target) windows
    Ch 3  attention.py               causal multi-head self-attention
    Ch 4  model.py, generate.py      GPT-2 architecture + decoding
    Ch 5  train.py, gpt2_weights.py  pretraining loop, loading OpenAI GPT-2 weights
    Ch 6  classifier.py, baselines.py  financial-sentiment classifier vs classical baselines
    Ch 7  instruct.py, lora.py       instruction fine-tuning (+ LoRA, Appendix E)
    Cap.  backtest/                  sentiment -> daily long/short signal -> backtest
"""

__version__ = "0.1.0"
