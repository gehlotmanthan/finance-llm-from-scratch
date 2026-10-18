"""Appendix E — Low-Rank Adaptation (LoRA).

Instead of updating a big weight matrix W (d_in x d_out), learn a low-rank
update ΔW = A·B with A (d_in x r) and B (r x d_out), r << d:

    y = x W + alpha · (x A B)

B starts at zero, so the model is unchanged at step 0. Only A and B train:
for GPT-2 medium with r = 16 that is roughly 2% of the parameters, so the
optimizer state is tiny -- the reason a 355M model can be instruction-tuned
on a 6 GB laptop GPU.
"""

import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model: nn.Module, rank: int, alpha: float, skip: tuple[str, ...] = ()) -> nn.Module:
    for name, module in model.named_children():
        if name in skip:
            continue
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha, skip)
    return model


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 16, skip=("out_head",)) -> tuple[int, int]:
    """Freeze the model, add LoRA adapters to every Linear (except ``skip``).
    Returns (trainable params, total params)."""
    for p in model.parameters():
        p.requires_grad = False
    replace_linear_with_lora(model, rank, alpha, skip)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
