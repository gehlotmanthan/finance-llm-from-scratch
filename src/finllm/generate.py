"""Turning next-token logits into text.

generate_text_simple   greedy decoding
generate               temperature scaling + top-k sampling + early stop on EOS
"""

import torch

from .tokenizer import EOS_TOKEN


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text, allowed_special={EOS_TOKEN})).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


@torch.no_grad()
def generate_text_simple(model, idx, max_new_tokens: int, context_size: int):
    """Append the arg-max token ``max_new_tokens`` times. ``idx``: (b, T) -> (b, T+new)."""
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


@torch.no_grad()
def generate(model, idx, max_new_tokens: int, context_size: int,
             temperature: float = 0.0, top_k: int | None = None, eos_id: int | None = None):
    """Temperature 0 = greedy; >1 flatter, <1 sharper. ``top_k`` keeps the k best logits."""
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < top_logits[:, -1:],
                                 torch.tensor(float("-inf"), device=logits.device), logits)
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx.shape[0] == 1 and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
