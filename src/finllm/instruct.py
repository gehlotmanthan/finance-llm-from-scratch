"""Instruction fine-tuning.

Prompt format (Alpaca style)::

    Below is an instruction that describes a task. Write a response that
    appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:            <- only if there is an input
    {input}

    ### Response:
    {output}

Batching: every example ends with one <|endoftext|>; the batch is padded
with more <|endoftext|> tokens, and every padding position *after the first*
gets target -100 so the loss ignores it. Optionally the prompt
tokens are masked too, so the model is only graded on the response.

Finance twist: alongside general finance Q&A (finance-alpaca) the training set
includes "classify the sentiment of this sentence" instructions built from the
sentiment classifier's *training* split, so the instruction-tuned model can be scored
on the very same PhraseBank test set as the classifier.
"""

import re

import torch
from torch.utils.data import Dataset

from .tokenizer import EOS_ID

PREAMBLE = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.")
SENTIMENT_INSTRUCTION = ("Classify the sentiment of this financial news sentence "
                         "as positive, negative or neutral.")
_LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b")


def format_input(entry: dict) -> str:
    text = f"{PREAMBLE}\n\n### Instruction:\n{entry['instruction']}"
    if entry.get("input"):
        text += f"\n\n### Input:\n{entry['input']}"
    return text


def response_prefix() -> str:
    return "\n\n### Response:\n"


def full_text(entry: dict) -> str:
    return format_input(entry) + response_prefix() + entry["output"]


class InstructionDataset(Dataset):
    """Pre-tokenised examples; each item is (token_ids, prompt_length)."""

    def __init__(self, data: list[dict], tokenizer):
        self.items = []
        for e in data:
            prompt = tokenizer.encode(format_input(e) + response_prefix())
            ids = tokenizer.encode(full_text(e))
            self.items.append((ids, len(prompt)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def custom_collate_fn(batch, pad_token_id: int = EOS_ID, ignore_index: int = -100,
                      allowed_max_length: int | None = None, device="cpu", mask_prompt: bool = False):
    batch_max = max(len(ids) + 1 for ids, _ in batch)
    inputs, targets = [], []
    for ids, prompt_len in batch:
        new = list(ids) + [pad_token_id]
        padded = new + [pad_token_id] * (batch_max - len(new))
        x = torch.tensor(padded[:-1])
        y = torch.tensor(padded[1:])
        pad_pos = torch.nonzero(y == pad_token_id).squeeze(-1)
        if pad_pos.numel() > 1:
            # keep the first end-of-text as a target (the model must learn to stop)
            first_eos = len(ids) - 1
            mask = torch.zeros_like(y, dtype=torch.bool)
            mask[first_eos + 1:] = True
            y[mask] = ignore_index
        if mask_prompt:
            y[:prompt_len - 1] = ignore_index
        if allowed_max_length is not None:
            x, y = x[:allowed_max_length], y[:allowed_max_length]
        inputs.append(x)
        targets.append(y)
    return torch.stack(inputs).to(device), torch.stack(targets).to(device)


def sentiment_instructions(df) -> list[dict]:
    """PhraseBank rows -> instruction examples (input = sentence, output = label word)."""
    return [{"instruction": SENTIMENT_INSTRUCTION, "input": t, "output": l}
            for t, l in zip(df["text"], df["label"])]


def parse_label(response: str) -> str | None:
    m = _LABEL_RE.search(response.lower())
    return m.group(1) if m else None


def extract_response(generated_text: str, prompt_text: str) -> str:
    """Text the model produced after the prompt, cut at the next '###' section."""
    resp = generated_text[len(prompt_text):] if generated_text.startswith(prompt_text) else generated_text
    return resp.split("###")[0].replace("<|endoftext|>", "").strip()


def token_length(entry: dict, tokenizer) -> int:
    return len(tokenizer.encode(full_text(entry)))
