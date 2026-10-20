"""Load the trained checkpoints and run them: used by the web app (app.py).

Every loader rebuilds the model from the config stored inside the checkpoint,
so the app needs nothing except the files in checkpoints/.
"""

from pathlib import Path

import numpy as np
import torch

from .classifier import GPTClassifier, predict_proba
from .generate import generate, text_to_token_ids, token_ids_to_text
from .instruct import extract_response, format_input, response_prefix
from .lora import apply_lora
from .model import GPTModel, get_attention_maps, set_attention_capture
from .backtest.signal import sentiment_score
from .tokenizer import EOS_ID, get_gpt2_tokenizer


def _load(path: Path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_language_model(path: Path, device="cpu") -> tuple[GPTModel, dict]:
    """A plain next-token model saved by pretrain_small_gpt.py or domain_adaptation.py."""
    ck = _load(path)
    model = GPTModel(ck["cfg"])
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), ck["cfg"]


def load_sentiment_model(path: Path, device="cpu"):
    """The classifier saved by sentiment_classifier.py. Returns (model, checkpoint dict)."""
    ck = _load(path)
    clf = GPTClassifier(GPTModel(ck["cfg"]), len(ck["labels"]))
    clf.load_state_dict(ck["model_state_dict"])
    return clf.to(device).eval(), ck


def load_instruct_model(path: Path, device="cpu") -> tuple[GPTModel, dict]:
    """The instruction-tuned model saved by instruction_tuning.py (with or without LoRA)."""
    ck = _load(path)
    model = GPTModel(ck["cfg"])
    if ck.get("lora"):
        apply_lora(model, ck["lora"]["rank"], ck["lora"]["alpha"])
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), ck


def classify(clf, ck, texts: list[str], device="cpu") -> list[dict]:
    """Sentiment probabilities, label and score P(pos) - P(neg) for each text."""
    probs = predict_proba(clf, texts, get_gpt2_tokenizer(), device, max_length=ck["max_length"])
    scores = sentiment_score(probs)
    labels = ck["labels"]
    return [{"text": t, **{labels[i]: float(p[i]) for i in range(len(labels))},
             "label": labels[int(np.argmax(p))], "score": float(s)}
            for t, p, s in zip(texts, probs, scores)]


def classify_baseline(pipe, texts: list[str], labels: list[str]) -> list[dict]:
    probs = pipe.predict_proba(texts)
    return [{"label": labels[int(np.argmax(p))], "score": float(p[2] - p[0])} for p in probs]


@torch.no_grad()
def complete(model, cfg, prompt: str, device="cpu", max_new_tokens: int = 40,
             temperature: float = 0.8, top_k: int | None = 40, seed: int | None = 123) -> str:
    """Continue a prompt. Returns only the newly generated text."""
    tok = get_gpt2_tokenizer()
    ids = text_to_token_ids(prompt, tok).to(device)
    if seed is not None:
        torch.manual_seed(seed)
    out = generate(model, ids, max_new_tokens, cfg["context_length"], temperature=temperature,
                   top_k=top_k, eos_id=EOS_ID)
    return token_ids_to_text(out[:, ids.shape[1]:].cpu(), tok)


@torch.no_grad()
def answer(model, cfg, instruction: str, input_text: str = "", device="cpu", max_new_tokens: int = 120,
           temperature: float = 0.0, top_k: int | None = None, seed: int | None = 123) -> str:
    """Ask the instruction-tuned model something, using the prompt format it was trained on."""
    tok = get_gpt2_tokenizer()
    prompt = format_input({"instruction": instruction, "input": input_text}) + response_prefix()
    ids = text_to_token_ids(prompt, tok).to(device)
    if seed is not None:
        torch.manual_seed(seed)
    out = generate(model, ids, max_new_tokens, cfg["context_length"], temperature=temperature,
                   top_k=top_k, eos_id=EOS_ID)
    return extract_response(token_ids_to_text(out.cpu(), tok), prompt)


@torch.no_grad()
def attention_maps(model, text: str, device="cpu") -> tuple[list[str], list[np.ndarray]]:
    """Tokens of ``text`` and, per layer, an array (heads, T, T) of attention weights."""
    tok = get_gpt2_tokenizer()
    ids = text_to_token_ids(text, tok).to(device)
    set_attention_capture(model, True)
    try:
        model(ids)
        maps = [m[0].float().cpu().numpy() for m in get_attention_maps(model)]
    finally:
        set_attention_capture(model, False)
    return [tok.decode([i]) for i in ids[0].tolist()], maps
