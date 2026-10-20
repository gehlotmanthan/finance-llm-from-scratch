"""Tokenizer, dataloader, attention and the GPT model."""

import pytest
import torch

from finllm.attention import CausalAttention, MultiHeadAttention, simplified_self_attention
from finllm.config import TINY_CONFIG, get_config, pretrained_config, resolve_name
from finllm.data import create_dataloader_v1, join_documents
from finllm.data_sources import load_the_verdict
from finllm.generate import generate, generate_text_simple
from finllm.model import (GPTModel, analytic_param_count, count_parameters,
                          get_attention_maps, set_attention_capture)
from finllm.tokenizer import (SimpleTokenizerV1, SimpleTokenizerV2, bpe_stats, build_vocab,
                              get_gpt2_tokenizer, preprocess_text, word_level_oov_rate, word_pieces)


# ----------------------------------------------------------------- tokenizer and data
def test_preprocess_and_simple_tokenizers_roundtrip():
    text = load_the_verdict()
    vocab = build_vocab(text)
    assert len(vocab) == 1132                       # 1,130 tokens + 2 specials
    tok = SimpleTokenizerV1(vocab)
    s = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    # The simple word-level decoder is lossy around quotes and apostrophes
    assert tok.decode(tok.encode(s)) == '" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    with pytest.raises(KeyError):
        tok.encode("EBITDA")
    tok2 = SimpleTokenizerV2(vocab)
    assert tok2.encode("EBITDA")[0] == vocab["<|unk|>"]
    assert preprocess_text("Hello, world.") == ["Hello", ",", "world", "."]


def test_gpt2_bpe_known_ids_and_pieces():
    enc = get_gpt2_tokenizer()
    assert enc.encode("Hello, do you like tea?") == [15496, 11, 466, 345, 588, 8887, 30]
    assert "".join(word_pieces("EBITDA")) == " EBITDA"


def test_bpe_stats_and_oov():
    s = bpe_stats(["profit rose", "EBITDA y/y"])
    assert s["n_words"] == 4 and s["tokens_per_word"] >= 1
    o = word_level_oov_rate(["profit rose"], ["profit fell"])
    assert o["oov_rate"] == pytest.approx(0.5)


def test_dataloader_targets_are_shifted_inputs():
    dl = create_dataloader_v1(load_the_verdict(), batch_size=2, max_length=8, stride=8, shuffle=False)
    x, y = next(iter(dl))
    assert x.shape == (2, 8) and torch.equal(x[:, 1:], y[:, :-1])
    assert x[0].tolist() == [40, 367, 2885, 1464, 1807, 3619, 402, 271]


def test_join_documents_uses_eos():
    assert join_documents(["a", " ", "b"]) == "a <|endoftext|> b"


# ----------------------------------------------------------------- attention
def test_simplified_attention_rows_are_convex_combinations():
    x = torch.tensor([[0.43, 0.15, 0.89], [0.55, 0.87, 0.66]])
    out = simplified_self_attention(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_mha_is_causal(use_sdpa):
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 16, 10, 0.0, 4, use_sdpa=use_sdpa).eval()
    x = torch.randn(2, 10, 16)
    x2 = x.clone()
    x2[:, 6:] += 1.0
    with torch.no_grad():
        assert torch.allclose(mha(x)[:, :6], mha(x2)[:, :6], atol=1e-6)
        assert not torch.allclose(mha(x)[:, 6:], mha(x2)[:, 6:])


def test_sdpa_matches_explicit_attention():
    torch.manual_seed(0)
    a = MultiHeadAttention(32, 32, 20, 0.0, 4, qkv_bias=True).eval()
    b = MultiHeadAttention(32, 32, 20, 0.0, 4, qkv_bias=True, use_sdpa=True).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 20, 32)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_single_head_mha_equals_causal_attention():
    torch.manual_seed(0)
    ca = CausalAttention(8, 8, 6, 0.0).eval()
    mha = MultiHeadAttention(8, 8, 6, 0.0, 1).eval()
    with torch.no_grad():
        for n in ("W_query", "W_key", "W_value"):
            getattr(mha, n).weight.copy_(getattr(ca, n).weight)
        mha.out_proj.weight.copy_(torch.eye(8))
        mha.out_proj.bias.zero_()
        x = torch.randn(2, 6, 8)
        assert torch.allclose(ca(x), mha(x), atol=1e-6)


# ----------------------------------------------------------------- GPT model
def test_gpt2_small_parameter_counts():
    model = GPTModel(get_config("small"))
    assert count_parameters(model) == 163_009_536
    assert count_parameters(model, tied=True) == 124_412_160
    assert analytic_param_count(get_config("small")) == 124_412_160
    assert analytic_param_count(pretrained_config("small")) == 124_439_808


def test_forward_shapes_and_generation():
    torch.manual_seed(0)
    m = GPTModel(TINY_CONFIG).eval()
    idx = torch.randint(0, 50257, (2, 5))
    assert m(idx).shape == (2, 5, 50257)
    out = generate_text_simple(m, idx, 4, TINY_CONFIG["context_length"])
    assert out.shape == (2, 9) and torch.equal(out[:, :5], idx)
    out2 = generate(m, idx[:1], 4, 64, temperature=1.0, top_k=5)
    assert out2.shape[1] <= 9


def test_attention_capture():
    m = GPTModel(TINY_CONFIG).eval()
    set_attention_capture(m, True)
    with torch.no_grad():
        m(torch.randint(0, 50257, (1, 7)))
    maps = get_attention_maps(m)
    assert len(maps) == TINY_CONFIG["n_layers"]
    assert maps[0].shape == (1, TINY_CONFIG["n_heads"], 7, 7)
    assert torch.allclose(maps[0].sum(-1), torch.ones(1, TINY_CONFIG["n_heads"], 7))
    assert torch.all(maps[0][0, 0].triu(1) == 0)          # nothing attends to the future


def test_config_aliases():
    assert resolve_name("medium") == "gpt2-medium (355M)"
    with pytest.raises(KeyError):
        resolve_name("gpt5")
