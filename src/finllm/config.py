"""Model configurations (Ch 4.1, Exercise 4.2, Ch 5.5).

A GPT model is fully described by one small dictionary:

    vocab_size      distinct token IDs (GPT-2 BPE = 50,257)
    context_length  max tokens the model sees at once
    emb_dim         width of each token vector
    n_heads         attention heads per layer
    n_layers        number of transformer blocks (depth)
    drop_rate       dropout probability
    qkv_bias        bias on the query/key/value projections (True for OpenAI weights)
    use_sdpa        use PyTorch's fused scaled_dot_product_attention (same maths,
                    less memory -- matters on an 8 GB GPU)
"""

import copy

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "use_sdpa": False,
}

_SIZE_OVERRIDES = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

GPT_CONFIGS = {name: {**GPT_CONFIG_124M, **o} for name, o in _SIZE_OVERRIDES.items()}

# Hugging Face repo names of OpenAI's released weights.
HF_REPOS = {
    "gpt2-small (124M)": "gpt2",
    "gpt2-medium (355M)": "gpt2-medium",
    "gpt2-large (774M)": "gpt2-large",
    "gpt2-xl (1558M)": "gpt2-xl",
}

_ALIASES = {
    "gpt2": "gpt2-small (124M)", "gpt2-small": "gpt2-small (124M)", "small": "gpt2-small (124M)",
    "124m": "gpt2-small (124M)",
    "gpt2-medium": "gpt2-medium (355M)", "medium": "gpt2-medium (355M)", "355m": "gpt2-medium (355M)",
    "gpt2-large": "gpt2-large (774M)", "large": "gpt2-large (774M)", "774m": "gpt2-large (774M)",
    "gpt2-xl": "gpt2-xl (1558M)", "xl": "gpt2-xl (1558M)", "1558m": "gpt2-xl (1558M)",
}

# A small model we can pretrain from scratch on a single consumer GPU in minutes
# (Ch 5). ~30M parameters, about half of them in the token embedding.
FIN_SMALL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "use_sdpa": True,
}

# Tiny model for unit tests and --smoke runs on CPU.
TINY_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "use_sdpa": False,
}


def resolve_name(name: str) -> str:
    if name in GPT_CONFIGS:
        return name
    key = name.lower().strip()
    if key in _ALIASES:
        return _ALIASES[key]
    raise KeyError(f"Unknown model {name!r}. Choose from {list(GPT_CONFIGS)} or {sorted(_ALIASES)}")


def get_config(name: str = "gpt2-small (124M)", **overrides) -> dict:
    """Copy of a named config with optional overrides, e.g. ``get_config('small', context_length=256)``."""
    cfg = copy.deepcopy(GPT_CONFIGS[resolve_name(name)])
    cfg.update(overrides)
    return cfg


def pretrained_config(name: str = "gpt2-small (124M)", **overrides) -> dict:
    """Config that matches OpenAI's checkpoints: 1024 context, bias on Q/K/V (Ch 5.5)."""
    cfg = get_config(name, context_length=1024, qkv_bias=True)
    cfg.update(overrides)
    return cfg
