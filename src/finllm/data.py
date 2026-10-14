"""Chapter 2.6 — sliding-window (input, target) pairs for next-token prediction.

For each window start i (stepping by ``stride``)::

    input  = ids[i     : i + max_length]
    target = ids[i + 1 : i + max_length + 1]     # the same window shifted by one

so target[t] is the token the model must predict after seeing input[:t+1].
"""

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import EOS_TOKEN, get_gpt2_tokenizer


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(txt, allowed_special={EOS_TOKEN})
        if len(token_ids) <= max_length:
            raise ValueError(
                f"Text has {len(token_ids)} tokens; need more than max_length={max_length}")
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt: str, batch_size: int = 4, max_length: int = 256,
                         stride: int = 128, shuffle: bool = True, drop_last: bool = True,
                         num_workers: int = 0, tokenizer=None) -> DataLoader:
    """Batches of (inputs, targets), each LongTensor ``(batch_size, max_length)``.

    ``num_workers=0`` keeps loading in the main process -- the safe choice on Windows.
    """
    tokenizer = tokenizer or get_gpt2_tokenizer()
    ds = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)


def join_documents(docs: list[str]) -> str:
    """Join independent documents (e.g. headlines) with GPT-2's end-of-text marker,
    so the model learns where one document stops and the next begins (2.4)."""
    return f" {EOS_TOKEN} ".join(d.strip() for d in docs if d and d.strip())


def train_val_split_text(text: str, train_ratio: float = 0.9) -> tuple[str, str]:
    """Split one long text at a character boundary (Ch 5.1.3)."""
    split = int(train_ratio * len(text))
    return text[:split], text[split:]
