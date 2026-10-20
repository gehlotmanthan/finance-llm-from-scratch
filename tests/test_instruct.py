"""Instruction tuning and LoRA: prompt format, collate masking, label parsing, LoRA."""

import torch

from finllm.config import TINY_CONFIG
from finllm.instruct import (InstructionDataset, custom_collate_fn, extract_response, format_input,
                             full_text, parse_label, response_prefix, sentiment_instructions)
from finllm.lora import LinearWithLoRA, apply_lora
from finllm.model import GPTModel
from finllm.tokenizer import get_gpt2_tokenizer


def test_format_input_with_and_without_input():
    a = format_input({"instruction": "Define EPS.", "input": ""})
    b = format_input({"instruction": "Classify.", "input": "Sales fell."})
    assert "### Input" not in a and a.endswith("Define EPS.")
    assert b.endswith("### Input:\nSales fell.")
    assert full_text({"instruction": "x", "input": "", "output": "y"}).endswith("### Response:\ny")


def test_collate_matches_book_example():
    # Book 7.3: inputs padded with 50256; targets shifted, extra padding -> -100
    batch = [([0, 1, 2, 3, 4], 1), ([5, 6], 1), ([7, 8, 9], 1)]
    x, y = custom_collate_fn(batch)
    assert x.tolist() == [[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]]
    assert y.tolist() == [[1, 2, 3, 4, 50256], [6, 50256, -100, -100, -100], [8, 9, 50256, -100, -100]]


def test_collate_prompt_masking_and_truncation():
    x, y = custom_collate_fn([([10, 11, 12, 13, 14], 3)], mask_prompt=True)
    assert y.tolist() == [[-100, -100, 13, 14, 50256]]
    x, y = custom_collate_fn([([10, 11, 12, 13, 14], 1)], allowed_max_length=3)
    assert x.shape == (1, 3)


def test_prompt_tokens_are_prefix_of_full_text():
    tok = get_gpt2_tokenizer()
    e = sentiment_instructions({"text": ["Operating profit rose to EUR 13.1 mn ."], "label": ["positive"]})[0]
    ids, plen = InstructionDataset([e], tok)[0]
    assert ids[:plen] == tok.encode(format_input(e) + response_prefix())
    assert tok.decode(ids[plen:]) == "positive"


def test_parse_and_extract():
    assert parse_label("Positive.") == "positive"
    assert parse_label("The sentiment is NEGATIVE") == "negative"
    assert parse_label("bullish") is None
    prompt = "P\n\n### Response:\n"
    assert extract_response(prompt + "neutral\n\n### Instruction: x", prompt) == "neutral"


def test_lora_starts_identical_and_trains_few_params():
    torch.manual_seed(0)
    m = GPTModel(TINY_CONFIG).eval()
    x = torch.randint(0, 50257, (1, 6))
    with torch.no_grad():
        before = m(x)
    trainable, total = apply_lora(m, rank=4, alpha=8)
    with torch.no_grad():
        after = m(x)
    assert torch.allclose(before, after)                      # B = 0 at init
    assert isinstance(m.trf_blocks[0].att.W_query, LinearWithLoRA)
    assert not isinstance(m.out_head, LinearWithLoRA)
    assert 0 < trainable < total * 0.2
    assert all(("lora" in n) == p.requires_grad for n, p in m.named_parameters())
