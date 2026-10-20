"""Round-trip the checkpoint loaders used by the web app, on tiny untrained models."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finllm import inference  # noqa: E402
from finllm.classifier import GPTClassifier  # noqa: E402
from finllm.config import TINY_CONFIG  # noqa: E402
from finllm.lora import apply_lora  # noqa: E402
from finllm.model import GPTModel  # noqa: E402

LABELS = ["negative", "neutral", "positive"]


def test_sentiment_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    clf = GPTClassifier(GPTModel(TINY_CONFIG), 3).eval()
    path = tmp_path / "sentiment.pth"
    torch.save({"model_state_dict": clf.state_dict(), "cfg": TINY_CONFIG, "labels": LABELS,
                "max_length": 32, "name": "tiny"}, path)
    loaded, ck = inference.load_sentiment_model(path)
    rows = inference.classify(loaded, ck, ["Profit rose", "Shares fell sharply after the warning"])
    assert len(rows) == 2
    for r in rows:
        probs = np.array([r[l] for l in LABELS])
        assert abs(probs.sum() - 1) < 1e-5
        assert r["label"] == LABELS[int(probs.argmax())]
        assert abs(r["score"] - (r["positive"] - r["negative"])) < 1e-5


def test_instruct_checkpoint_with_lora_round_trip(tmp_path):
    torch.manual_seed(0)
    model = GPTModel(TINY_CONFIG)
    apply_lora(model, 4, 4)
    path = tmp_path / "instruct.pth"
    torch.save({"model_state_dict": model.state_dict(), "cfg": TINY_CONFIG,
                "lora": {"rank": 4, "alpha": 4}}, path)
    loaded, ck = inference.load_instruct_model(path)
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    out = inference.answer(loaded, ck["cfg"], "What is a dividend?", max_new_tokens=5)
    assert isinstance(out, str)


def test_complete_and_attention_maps(tmp_path):
    torch.manual_seed(0)
    model = GPTModel(TINY_CONFIG)
    path = tmp_path / "lm.pth"
    torch.save({"model_state_dict": model.state_dict(), "cfg": TINY_CONFIG}, path)
    loaded, cfg = inference.load_language_model(path)
    a = inference.complete(loaded, cfg, "Shares rose", max_new_tokens=6, seed=1)
    b = inference.complete(loaded, cfg, "Shares rose", max_new_tokens=6, seed=1)
    assert a == b  # seeded sampling is reproducible

    tokens, maps = inference.attention_maps(loaded, "Tesla shares fall")
    assert len(maps) == TINY_CONFIG["n_layers"]
    T = len(tokens)
    for m in maps:
        assert m.shape == (TINY_CONFIG["n_heads"], T, T)
        assert np.allclose(m.sum(-1), 1, atol=1e-5)  # rows are distributions
        assert np.allclose(np.triu(m, k=1), 0)        # no attention to future tokens
