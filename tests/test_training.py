"""Training loop, perplexity, GPT-2 weight mapping, checkpoints."""

import math

import pytest
import torch

from finllm.config import TINY_CONFIG
from finllm.data import create_dataloader_v1
from finllm.data_sources import load_the_verdict
from finllm.gpt2_weights import gpt_to_hf_params, load_weights_into_gpt
from finllm.model import GPTModel
from finllm.tokenizer import get_gpt2_tokenizer
from finllm.train import (calc_loss_loader, corpus_loss, load_checkpoint, lr_at_step, perplexity,
                          save_checkpoint, train_lm, unigram_baseline_loss)


def test_untrained_loss_is_near_uniform():
    torch.manual_seed(0)
    m = GPTModel(TINY_CONFIG)
    dl = create_dataloader_v1(load_the_verdict(), batch_size=2, max_length=16, stride=16, shuffle=False)
    loss = calc_loss_loader(dl, m, "cpu", num_batches=3)
    assert abs(loss - math.log(50257)) < 0.5      # ~10.8 = log(vocab): a uniform guess


def test_training_reduces_loss():
    torch.manual_seed(0)
    m = GPTModel(TINY_CONFIG)
    text = load_the_verdict()
    dl = create_dataloader_v1(text[:4000], batch_size=4, max_length=16, stride=16, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    hist = train_lm(m, dl, dl, opt, torch.device("cpu"), num_epochs=3, eval_freq=5, eval_iter=2,
                    max_lr=5e-3, warmup_steps=2, grad_clip=1.0, grad_accum=2)
    assert hist["train_loss"][-1] < hist["train_loss"][0] - 1.0


def test_lr_schedule_shape():
    lrs = [lr_at_step(s, 100, 1.0, warmup_steps=10, min_lr=0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and max(lrs) == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1, abs=1e-3)
    assert all(a <= b for a, b in zip(lrs[:10], lrs[1:10]))


def test_perplexity_and_corpus_loss():
    assert perplexity(0.0) == 1.0
    m = GPTModel(TINY_CONFIG)
    r = corpus_loss(m, load_the_verdict()[:3000], get_gpt2_tokenizer(), "cpu", max_length=32)
    assert r["perplexity"] == pytest.approx(math.exp(r["loss"]))


def test_unigram_baseline():
    # val tokens identical to the most common training token -> low loss
    assert unigram_baseline_loss([5] * 1000, [5] * 10) < unigram_baseline_loss([5] * 1000, [7] * 10)


@pytest.mark.parametrize("prefix", ["", "transformer."])
def test_gpt2_weight_mapping_roundtrip(prefix):
    torch.manual_seed(1)
    src = GPTModel(TINY_CONFIG).eval()
    params = {prefix + k: v for k, v in gpt_to_hf_params(src).items()}
    params = {k.removeprefix("transformer."): v for k, v in params.items()}
    torch.manual_seed(2)
    dst = GPTModel(TINY_CONFIG).eval()
    load_weights_into_gpt(dst, params)
    x = torch.randint(0, 50257, (2, 9))
    with torch.no_grad():
        # out_head is tied to wte in GPT-2, so compare against a tied source
        src.out_head.weight = torch.nn.Parameter(src.tok_emb.weight.clone())
        assert torch.allclose(src(x), dst(x), atol=1e-5)


def test_weight_loading_requires_qkv_bias():
    src = GPTModel(TINY_CONFIG)
    dst = GPTModel(dict(TINY_CONFIG, qkv_bias=False))
    with pytest.raises(ValueError):
        load_weights_into_gpt(dst, gpt_to_hf_params(src))


def test_checkpoint_roundtrip(tmp_path):
    m = GPTModel(TINY_CONFIG)
    p = save_checkpoint(tmp_path / "m.pth", m, TINY_CONFIG, note="x")
    ck = load_checkpoint(p)
    m2 = GPTModel(ck["cfg"])
    m2.load_state_dict(ck["model_state_dict"])
    assert ck["note"] == "x"
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
