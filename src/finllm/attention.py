"""Chapter 3 — attention, built up in four steps.

    3.3  simplified_self_attention   no trainable weights
    3.4  SelfAttention_v2            trainable W_q, W_k, W_v
    3.5  CausalAttention             + causal mask + dropout
    3.6  MultiHeadAttention          h heads in one batched matmul (used by GPTModel)

    context = softmax( Q Kᵀ / sqrt(d_k) + causal_mask ) V

Additions over the book, both switchable and covered by tests:

* ``use_sdpa`` — call ``torch.nn.functional.scaled_dot_product_attention``
  (fused / flash kernels). Mathematically identical, never materialises the
  T x T matrix, so long contexts fit on a small (6-8 GB) GPU.
* ``store_attn`` — keep the last attention weights for plotting
  (scripts/ch05_gpt2_domain.py shows what GPT-2 attends to in a headline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def simplified_self_attention(inputs: torch.Tensor) -> torch.Tensor:
    """3.3: context vectors as similarity-weighted averages of the inputs. ``(T, d) -> (T, d)``."""
    attn_weights = torch.softmax(inputs @ inputs.T, dim=-1)
    return attn_weights @ inputs


class SelfAttention_v2(nn.Module):
    """3.4: single sequence ``(T, d_in) -> (T, d_out)`` with trainable projections."""

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        q, k, v = self.W_query(x), self.W_key(x), self.W_value(x)
        w = torch.softmax(q @ k.T / k.shape[-1] ** 0.5, dim=-1)
        return w @ v


class CausalAttention(nn.Module):
    """3.5: batched single-head attention that cannot look at future tokens."""

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, T, _ = x.shape
        q, k, v = self.W_query(x), self.W_key(x), self.W_value(x)
        scores = q @ k.transpose(1, 2)
        scores.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        w = self.dropout(torch.softmax(scores / k.shape[-1] ** 0.5, dim=-1))
        return w @ v


class MultiHeadAttention(nn.Module):
    """3.6.2: efficient multi-head causal self-attention.

    Shapes (b batch, T tokens, h heads, hd = d_out // h)::

        x (b,T,d_in) -> Q,K,V (b,T,d_out) -> view (b,T,h,hd) -> transpose (b,h,T,hd)
        Q Kᵀ (b,h,T,T) + mask -> softmax -> dropout -> @ V (b,h,T,hd)
        -> transpose/reshape (b,T,d_out) -> out_proj
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False, use_sdpa=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.use_sdpa = use_sdpa
        self.store_attn = False
        self.last_attn = None  # (b, h, T, T) when store_attn is True

    def forward(self, x):
        b, T, _ = x.shape
        q = self.W_query(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_sdpa and not self.store_attn:
            ctx = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        else:
            scores = q @ k.transpose(2, 3)
            scores.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
            w = torch.softmax(scores / k.shape[-1] ** 0.5, dim=-1)
            if self.store_attn:
                self.last_attn = w.detach()
            ctx = self.dropout(w) @ v

        ctx = ctx.transpose(1, 2).reshape(b, T, self.d_out)
        return self.out_proj(ctx)
