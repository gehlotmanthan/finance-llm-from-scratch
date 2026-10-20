"""The GPT-2 architecture.

    token IDs (b,T)
      -> tok_emb + pos_emb -> dropout                      (b,T,D)
      -> n_layers x TransformerBlock                       (b,T,D)
           x = x + Drop(MHA(LN(x)))      # attention sub-layer, pre-LayerNorm
           x = x + Drop(FFN(LN(x)))      # feed-forward sub-layer
      -> final LayerNorm -> out_head (D -> V)              (b,T,V) logits

Parameter names follow the common from-scratch GPT-2 layout (tok_emb, trf_blocks,
out_head, ...), which keeps checkpoints simple to map to and from OpenAI's weights.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """per-token normalisation with trainable scale/shift.
    ``unbiased=False`` matches GPT-2's TensorFlow original (needed for its weights)."""

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    """tanh-approximated GELU, as used by GPT-2."""

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """position-wise MLP D -> 4D -> D."""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """pre-LN block with two residual ("shortcut") connections."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"],
            use_sdpa=cfg.get("use_sdpa", False))
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """full GPT. ``forward(in_idx (b,T)) -> logits (b,T,vocab)``.

    GPT_CONFIG_124M: 163,009,536 parameters, or 124,412,160 with the output
    head tied to the token embedding (how OpenAI shipped GPT-2).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, T = in_idx.shape
        x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(T, device=in_idx.device))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


# ---------------------------------------------------------------------------
# Accounting helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, tied: bool = False, trainable_only: bool = False) -> int:
    params = [p for p in model.parameters() if p.requires_grad or not trainable_only]
    total = sum(p.numel() for p in params)
    if tied:
        total -= sum(p.numel() for p in model.out_head.parameters())
    return total


def model_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    return sum(p.numel() for p in model.parameters()) * bytes_per_param / 1024 ** 2


def analytic_param_count(cfg: dict, tied: bool = True) -> int:
    """Closed-form parameter count — checked against the real model in tests.

    per block: attention 3·D² (+3D if qkv_bias) + D² + D, FFN 8·D² + 5D, 2 LayerNorms 4D
    """
    D, V, C, L = cfg["emb_dim"], cfg["vocab_size"], cfg["context_length"], cfg["n_layers"]
    attn = 3 * D * D + (3 * D if cfg["qkv_bias"] else 0) + D * D + D
    ffn = D * 4 * D + 4 * D + 4 * D * D + D
    block = attn + ffn + 4 * D
    total = V * D + C * D + L * block + 2 * D
    if not tied:
        total += V * D
    return total


def training_flops_per_token(cfg: dict) -> float:
    """≈ 6·N (+ attention term) FLOPs per token for forward + backward (Kaplan et al., 2020)."""
    n = analytic_param_count(cfg, tied=True) - cfg["vocab_size"] * cfg["emb_dim"] - cfg["context_length"] * cfg["emb_dim"]
    attn = 2 * cfg["n_layers"] * cfg["context_length"] * cfg["emb_dim"]
    return 6 * n + 3 * attn


def set_attention_capture(model: nn.Module, on: bool = True) -> None:
    """Make every attention layer keep its weights (for plots). Disables SDPA while on."""
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            m.store_attn = on
            if not on:
                m.last_attn = None


def get_attention_maps(model: nn.Module) -> list[torch.Tensor]:
    """List (one per layer) of (b, h, T, T) attention weights from the last forward pass."""
    return [m.last_attn for m in model.modules() if isinstance(m, MultiHeadAttention)]
