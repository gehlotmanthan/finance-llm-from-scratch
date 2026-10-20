"""From raw text to token IDs, plus tools to measure how well a
tokenizer fits *financial* text.

    preprocess_text / build_vocab / SimpleTokenizerV1/V2   word-level tokenizers
    get_gpt2_tokenizer                                     GPT-2 byte-pair encoding
    bpe_stats, word_pieces, word_level_oov_rate            analysis used in scripts/tokenizer_study.py

Why it matters for finance: a word-level vocabulary built on one corpus breaks
on the next (new tickers, numbers, "Q3", "y/y"). BPE never fails -- it falls
back to sub-word pieces -- but fragmenting "EBITDA" into 3 tokens costs context
length and makes the model learn the term from pieces.
"""

import re
from functools import lru_cache

import tiktoken

SPLIT_PATTERN = r'([,.:;?_!"()\']|--|\s)'
JOIN_PATTERN = r'\s+([,.?!"()\'])'
EOS_TOKEN = "<|endoftext|>"
EOS_ID = 50256


def preprocess_text(text: str) -> list[str]:
    """Split text into words and punctuation.

    >>> preprocess_text("Hello, world. Is this-- a test?")
    ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
    """
    return [t.strip() for t in re.split(SPLIT_PATTERN, text) if t.strip()]


def build_vocab(text: str, special_tokens=(EOS_TOKEN, "<|unk|>")) -> dict[str, int]:
    """Sorted unique tokens -> integer IDs, special tokens appended last."""
    tokens = sorted(set(preprocess_text(text)))
    tokens.extend(special_tokens)
    return {tok: i for i, tok in enumerate(tokens)}


class SimpleTokenizerV1:
    """Word-level tokenizer. Raises KeyError on unseen words."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int[s] for s in preprocess_text(text)]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(JOIN_PATTERN, r"\1", text)


class SimpleTokenizerV2(SimpleTokenizerV1):
    """V1 + unknown words mapped to <|unk|>."""

    def __init__(self, vocab: dict[str, int]):
        if "<|unk|>" not in vocab:
            raise ValueError('SimpleTokenizerV2 needs "<|unk|>" in the vocabulary')
        super().__init__(vocab)

    def encode(self, text: str) -> list[int]:
        toks = [t if t in self.str_to_int else "<|unk|>" for t in preprocess_text(text)]
        return [self.str_to_int[t] for t in toks]


@lru_cache(maxsize=1)
def get_gpt2_tokenizer():
    """OpenAI's GPT-2 BPE tokenizer (50,257 tokens) via tiktoken."""
    return tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# Analysis helpers (used by scripts/tokenizer_study.py)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def bpe_stats(texts: list[str], enc=None) -> dict:
    """How efficiently does BPE encode this corpus?

    Returns tokens per word, characters per token and the share of
    whitespace-separated words that need more than one token. Words are
    encoded with a leading space, which is how they appear mid-sentence.
    """
    enc = enc or get_gpt2_tokenizer()
    n_words = n_tokens_words = n_split = n_chars = n_tokens_text = 0
    for text in texts:
        n_chars += len(text)
        n_tokens_text += len(enc.encode(text, allowed_special={EOS_TOKEN}))
        for w in _WORD_RE.findall(text):
            k = len(enc.encode(" " + w))
            n_words += 1
            n_tokens_words += k
            n_split += k > 1
    return {
        "n_texts": len(texts),
        "n_words": n_words,
        "n_tokens": n_tokens_text,
        "tokens_per_word": n_tokens_words / max(n_words, 1),
        "chars_per_token": n_chars / max(n_tokens_text, 1),
        "share_words_split": n_split / max(n_words, 1),
    }


def word_pieces(term: str, enc=None, leading_space: bool = True) -> list[str]:
    """The sub-word pieces BPE splits ``term`` into, e.g. ' EBITDA' -> [' EB', 'IT', 'DA']."""
    enc = enc or get_gpt2_tokenizer()
    ids = enc.encode((" " if leading_space else "") + term)
    return [enc.decode([i]) for i in ids]


def word_level_oov_rate(train_texts: list[str], test_texts: list[str]) -> dict:
    """Out-of-vocabulary rate of a word-level tokenizer built on ``train_texts``.

    This is the failure mode that motivates BPE: with a closed vocabulary every
    new ticker or number becomes <|unk|>.
    """
    vocab = build_vocab(" ".join(train_texts))
    tok = SimpleTokenizerV2(vocab)
    unk = vocab["<|unk|>"]
    total = n_unk = 0
    for t in test_texts:
        ids = tok.encode(t)
        total += len(ids)
        n_unk += sum(i == unk for i in ids)
    return {"vocab_size": len(vocab), "test_tokens": total, "unk_tokens": n_unk,
            "oov_rate": n_unk / max(total, 1)}
