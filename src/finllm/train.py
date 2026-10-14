"""Chapter 5 — pretraining: loss, evaluation, and the training loop.

Loss: cross-entropy between the model's next-token distribution and the true
next token, averaged over every position in the batch (5.1.2).
Perplexity = exp(loss) -- "the model is as unsure as a uniform choice among
exp(loss) tokens".

``train_lm`` is the book's ``train_model_simple`` plus the Appendix D upgrades,
all off by default so the plain book loop is still what runs:

* linear warm-up then cosine decay of the learning rate
* gradient clipping
* gradient accumulation (bigger effective batch on an 8 GB GPU)
* bfloat16 autocast on GPUs that support it
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .generate import generate_text_simple, text_to_token_ids, token_ids_to_text
from .utils import bf16_supported


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches: int | None = None) -> float:
    if len(data_loader) == 0:
        return float("nan")
    n = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total = 0.0
    for i, (x, y) in enumerate(data_loader):
        if i >= n:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def perplexity(loss: float) -> float:
    return float(math.exp(loss)) if loss < 50 else float("inf")


@torch.no_grad()
def corpus_loss(model, text: str, tokenizer, device, max_length: int = 256,
                batch_size: int = 8, max_windows: int | None = None) -> dict:
    """Token-weighted cross-entropy on non-overlapping windows of ``text`` -> loss & perplexity.

    Used to compare how 'surprised' a model is by different kinds of text
    (finance vs general English) and how that changes with domain training.
    """
    from .data import create_dataloader_v1
    dl = create_dataloader_v1(text, batch_size=batch_size, max_length=max_length, stride=max_length,
                              shuffle=False, drop_last=False, tokenizer=tokenizer)
    model.eval()
    total, n_tok = 0.0, 0
    for i, (x, y) in enumerate(dl):
        if max_windows is not None and i * batch_size >= max_windows:
            break
        logits = model(x.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten(), reduction="sum")
        total += loss.item()
        n_tok += y.numel()
    mean = total / max(n_tok, 1)
    return {"loss": mean, "perplexity": perplexity(mean), "tokens": n_tok}


def unigram_baseline_loss(train_ids: list[int], val_ids: list[int], vocab_size: int = 50257) -> float:
    """Cross-entropy of a context-free word-frequency model (add-one smoothing).
    A trained LM must beat this, or it has learned nothing beyond token counts."""
    counts = torch.bincount(torch.tensor(train_ids), minlength=vocab_size).double() + 1.0
    logp = torch.log(counts / counts.sum())
    return float(-logp[torch.tensor(val_ids)].mean())


def lr_at_step(step: int, total_steps: int, max_lr: float, warmup_steps: int = 0,
               min_lr: float = 0.0) -> float:
    """Appendix D.1-D.2: linear warm-up to ``max_lr`` then cosine decay to ``min_lr``."""
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_lm(model, train_loader, val_loader, optimizer, device, num_epochs: int,
             eval_freq: int, eval_iter: int, start_context: str | None = None,
             tokenizer=None, max_lr: float | None = None, warmup_steps: int = 0,
             min_lr: float = 0.0, grad_clip: float | None = None, grad_accum: int = 1,
             amp: bool = False, max_steps: int | None = None, log=print) -> dict:
    """Next-token training loop. Returns a history dict (losses, tokens seen, lrs)."""
    hist = {"step": [], "tokens_seen": [], "train_loss": [], "val_loss": [], "lr": [], "samples": []}
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max_steps or steps_per_epoch * num_epochs
    use_amp = amp and bf16_supported(device)
    tokens_seen, step = 0, 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = calc_loss_batch(x, y, model, device) / grad_accum
            loss.backward()
            tokens_seen += x.numel()
            if (i + 1) % grad_accum != 0 and (i + 1) != len(train_loader):
                continue

            if max_lr is not None:
                lr = lr_at_step(step, total_steps, max_lr, warmup_steps, min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % eval_freq == 0 or step == total_steps:
                tr, va = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                hist["step"].append(step)
                hist["tokens_seen"].append(tokens_seen)
                hist["train_loss"].append(tr)
                hist["val_loss"].append(va)
                hist["lr"].append(optimizer.param_groups[0]["lr"])
                log(f"ep {epoch + 1} step {step:>6}/{total_steps}  train {tr:.3f}  val {va:.3f}  "
                    f"ppl {perplexity(va):.1f}")
            if max_steps is not None and step >= max_steps:
                break
        if start_context and tokenizer is not None:
            hist["samples"].append({"epoch": epoch + 1,
                                    "text": sample_text(model, tokenizer, device, start_context)})
            log(f"  sample: {hist['samples'][-1]['text']!r}")
        if max_steps is not None and step >= max_steps:
            break
    hist["total_steps"] = step
    hist["total_tokens"] = tokens_seen
    return hist


def sample_text(model, tokenizer, device, start_context: str, max_new_tokens: int = 30) -> str:
    model.eval()
    ctx = model.pos_emb.weight.shape[0]
    ids = text_to_token_ids(start_context, tokenizer).to(device)
    out = generate_text_simple(model, ids, max_new_tokens=max_new_tokens, context_size=ctx)
    model.train()
    return token_ids_to_text(out, tokenizer).replace("\n", " ")


def save_checkpoint(path: Path, model, cfg: dict, optimizer=None, **extra) -> Path:
    """5.4: save weights (+ optimizer state to resume) and the config needed to rebuild."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "cfg": cfg, **extra}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
