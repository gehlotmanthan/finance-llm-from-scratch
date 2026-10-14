"""Chapter 6 — fine-tuning GPT for classification (financial sentiment).

The book's recipe, applied to 3-class financial sentiment instead of spam:

    1. replace the 50,257-way output head with a 3-way head (negative/neutral/positive);
    2. freeze most of the network, train the last block + final LayerNorm + head (6.5);
    3. classify from the output at the *last* token -- with causal attention it is
       the only position that has seen the whole sentence (6.6).

One change from the book: the book pads every text to the same length and
reads position -1, which for short texts is a padding token. Here the head
reads the last *real* token (``pool="last_real"``), and ``predict_proba`` pads
each batch only to its own longest text, which makes scoring a million
headlines in the capstone fast.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .stats import macro_f1
from .tokenizer import EOS_ID
from .utils import bf16_supported


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int | None = None, pad_token_id: int = EOS_ID):
        enc = [tokenizer.encode(t) for t in texts]
        self.max_length = max_length or max(len(e) for e in enc)
        enc = [e[:self.max_length] for e in enc]
        self.lengths = [max(len(e), 1) for e in enc]
        self.ids = [e + [pad_token_id] * (self.max_length - len(e)) for e in enc]
        self.labels = list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return (torch.tensor(self.ids[i]), torch.tensor(self.labels[i]), torch.tensor(self.lengths[i]))


class GPTClassifier(nn.Module):
    """A GPTModel whose output head predicts ``num_classes`` labels."""

    def __init__(self, gpt: nn.Module, num_classes: int = 3, pool: str = "last_real"):
        super().__init__()
        self.gpt = gpt
        self.gpt.out_head = nn.Linear(gpt.out_head.in_features, num_classes)
        self.pool = pool

    def forward(self, ids, lengths=None):
        logits = self.gpt(ids)                                   # (b, T, C)
        if self.pool == "last" or lengths is None:
            return logits[:, -1, :]
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        return logits.gather(1, idx).squeeze(1)                  # (b, C)


def set_trainable(clf: GPTClassifier, mode: str = "last_block") -> int:
    """'head' | 'last_block' (book default) | 'all'. Returns # trainable parameters."""
    g = clf.gpt
    for p in g.parameters():
        p.requires_grad = mode == "all"
    if mode in ("head", "last_block"):
        for p in g.out_head.parameters():
            p.requires_grad = True
    if mode == "last_block":
        for p in list(g.trf_blocks[-1].parameters()) + list(g.final_norm.parameters()):
            p.requires_grad = True
    if mode not in ("head", "last_block", "all"):
        raise ValueError(mode)
    return sum(p.numel() for p in clf.parameters() if p.requires_grad)


@torch.no_grad()
def predict_logits_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    out, ys = [], []
    for ids, y, lengths in loader:
        out.append(model(ids.to(device), lengths.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs: int = 5,
                     class_weights=None, amp: bool = True, log=print) -> dict:
    """Train with cross-entropy; after each epoch score validation macro-F1 and keep the best weights."""
    use_amp = amp and bf16_supported(device)
    weight = torch.tensor(class_weights, dtype=torch.float, device=device) if class_weights is not None else None
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best = (-1.0, None, 0)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ids, y, lengths in train_loader:
            ids, y, lengths = ids.to(device), y.to(device), lengths.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, lengths)
            loss = nn.functional.cross_entropy(logits.float(), y, weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        logits, yv = predict_logits_loader(model, val_loader, device)
        vloss = float(nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(yv)))
        pred = logits.argmax(1)
        f1 = macro_f1(yv, pred, logits.shape[1])
        hist["epoch"].append(epoch)
        hist["train_loss"].append(total / max(n, 1))
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(float((pred == yv).mean()))
        hist["val_macro_f1"].append(f1)
        log(f"  epoch {epoch}: train loss {total / max(n, 1):.3f}  val loss {vloss:.3f}  "
            f"val acc {hist['val_acc'][-1]:.3f}  val macro-F1 {f1:.3f}")
        if f1 > best[0]:
            best = (f1, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), epoch)
    model.load_state_dict(best[1])
    hist["best_epoch"] = best[2]
    hist["best_val_macro_f1"] = best[0]
    return hist


@torch.no_grad()
def predict_proba(model, texts, tokenizer, device, max_length: int = 128, batch_size: int = 256,
                  pad_token_id: int = EOS_ID, show_progress: bool = False) -> np.ndarray:
    """Class probabilities for a list of texts, batched by length (dynamic padding)."""
    model.eval()
    enc = [tokenizer.encode(t)[:max_length] or [pad_token_id] for t in texts]
    order = np.argsort([len(e) for e in enc], kind="stable")
    probs = np.zeros((len(texts), model.gpt.out_head.out_features), dtype=np.float32)
    use_amp = bf16_supported(torch.device(device) if isinstance(device, str) else device)
    starts = range(0, len(order), batch_size)
    if show_progress:
        from tqdm import tqdm
        starts = tqdm(starts, desc="scoring", unit="batch")
    for s in starts:
        idx = order[s:s + batch_size]
        L = max(len(enc[i]) for i in idx)
        ids = torch.tensor([enc[i] + [pad_token_id] * (L - len(enc[i])) for i in idx], device=device)
        lengths = torch.tensor([len(enc[i]) for i in idx], device=device)
        with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids, lengths)
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def make_loaders(splits: dict, tokenizer, batch_size: int = 8, max_length: int | None = None, seed: int = 123):
    """DataLoaders for train/val/test; max length = longest training text (capped at 128)."""
    train = SentimentDataset(splits["train"]["text"].tolist(), splits["train"]["label_id"].tolist(), tokenizer,
                             max_length=max_length)
    L = min(train.max_length, 128)
    if L != train.max_length:
        train = SentimentDataset(splits["train"]["text"].tolist(), splits["train"]["label_id"].tolist(), tokenizer, L)
    ds = {"train": train}
    for k in ("val", "test"):
        ds[k] = SentimentDataset(splits[k]["text"].tolist(), splits[k]["label_id"].tolist(), tokenizer, L)
    g = torch.Generator().manual_seed(seed)
    return ({"train": DataLoader(ds["train"], batch_size, shuffle=True, drop_last=True, generator=g),
             "val": DataLoader(ds["val"], batch_size * 4),
             "test": DataLoader(ds["test"], batch_size * 4)}, L)
