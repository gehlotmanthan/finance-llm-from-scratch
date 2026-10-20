"""Sentiment classifier: data parsing/splitting, classifier pooling, scoring, baselines, statistics."""

import io
import zipfile

import numpy as np
import pytest
import torch

from finllm import data_sources as ds
from finllm import synthetic
from finllm.baselines import fit_tfidf_lr, majority_predict
from finllm.classifier import GPTClassifier, SentimentDataset, predict_proba, set_trainable, train_classifier
from finllm.config import TINY_CONFIG
from finllm.model import GPTModel
from finllm.stats import (bootstrap_ci, classification_metrics, macro_f1, mcnemar_exact,
                          paired_bootstrap_diff)
from finllm.tokenizer import get_gpt2_tokenizer


def _fake_zip():
    files = {
        "Sentences_50Agree.txt": "Profit rose .@positive\nSales fell .@negative\nBoard met .@neutral\n"
                                 "Odd case .@positive\nOdd case .@negative\nBoard met .@neutral\nNo label here\n",
        "Sentences_AllAgree.txt": "Profit rose .@positive\n",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(f"FinancialPhraseBank-v1.0/{name}", text.encode("latin-1"))
    return buf.getvalue()


def test_phrasebank_parser_dedupes_and_drops_conflicts():
    df = ds.parse_phrasebank_zip(_fake_zip())
    assert set(df["text"]) == {"Profit rose .", "Sales fell .", "Board met ."}
    assert df.loc[df.text == "Profit rose .", "agreement"].item() == 100
    assert df.attrs["n_conflicting_dropped"] == 1
    assert df["label_id"].tolist() == [ds.LABEL2ID[l] for l in df["label"]]


def test_split_is_stratified_disjoint_and_deterministic():
    df = synthetic.phrasebank_like(500)
    a = ds.split_phrasebank(df, seed=1)
    b = ds.split_phrasebank(df, seed=1)
    assert sum(len(v) for v in a.values()) == 500
    assert not set(a["train"].text) & set(a["test"].text)
    assert a["test"].text.tolist() == b["test"].text.tolist()
    share = lambda d: d["label_id"].value_counts(normalize=True).sort_index().to_numpy()
    assert np.allclose(share(a["train"]), share(a["test"]), atol=0.05)


def test_dataset_padding_and_lengths():
    tok = get_gpt2_tokenizer()
    d = SentimentDataset(["profit rose", "a much longer sentence about falling sales"], [2, 0], tok)
    ids, y, L = d[0]
    assert len(ids) == d.max_length and L.item() == 2 and ids[2].item() == 50256


def test_last_real_pooling_and_dynamic_padding_match():
    torch.manual_seed(0)
    tok = get_gpt2_tokenizer()
    clf = GPTClassifier(GPTModel(TINY_CONFIG), 3).eval()
    texts = ["profit rose", "a much longer sentence about falling sales and margins", "flat"]
    d = SentimentDataset(texts, [0, 0, 0], tok)
    ids = torch.stack([d[i][0] for i in range(3)])
    lengths = torch.stack([d[i][2] for i in range(3)])
    with torch.no_grad():
        fixed = torch.softmax(clf(ids, lengths), -1).numpy()
        # each text alone, unpadded: the reference answer
        alone = np.stack([torch.softmax(clf(torch.tensor([tok.encode(t)]), None), -1)[0].numpy() for t in texts])
    dyn = predict_proba(clf, texts, tok, "cpu", batch_size=2)
    assert np.allclose(fixed, alone, atol=1e-5)
    assert np.allclose(dyn, alone, atol=1e-5)


def test_set_trainable_modes():
    clf = GPTClassifier(GPTModel(TINY_CONFIG), 3)
    n_head = set_trainable(clf, "head")
    n_block = set_trainable(clf, "last_block")
    n_all = set_trainable(clf, "all")
    assert n_head == TINY_CONFIG["emb_dim"] * 3 + 3
    assert n_head < n_block < n_all
    with pytest.raises(ValueError):
        set_trainable(clf, "everything")


def test_classifier_learns_synthetic_task():
    torch.manual_seed(0)
    tok = get_gpt2_tokenizer()
    df = synthetic.phrasebank_like(240, seed=3)
    sp = ds.split_phrasebank(df, seed=0)
    from finllm.classifier import make_loaders
    loaders, _ = make_loaders(sp, tok, batch_size=8)
    clf = GPTClassifier(GPTModel(TINY_CONFIG), 3)
    set_trainable(clf, "all")
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-3)
    hist = train_classifier(clf, loaders["train"], loaders["val"], opt, torch.device("cpu"), num_epochs=4, log=lambda *_: None)
    assert hist["best_val_macro_f1"] > 0.6


def test_baselines():
    df = synthetic.phrasebank_like(400, seed=1)
    sp = ds.split_phrasebank(df, seed=0)
    pipe, info = fit_tfidf_lr(sp["train"].text, sp["train"].label_id, sp["val"].text, sp["val"].label_id)
    assert macro_f1(sp["test"].label_id.to_numpy(), pipe.predict(sp["test"].text)) > 0.9
    assert (majority_predict([1, 1, 2], 4) == 1).all()


def test_classification_stats():
    y = np.array([0, 1, 2, 1, 1, 0])
    p = np.array([0, 1, 1, 1, 2, 0])
    m = classification_metrics(y, p, ["neg", "neu", "pos"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["confusion_matrix"][2][1] == 1
    lo, hi = bootstrap_ci(lambda a, b: float((a == b).mean()), y, p, n_boot=300)
    assert lo <= m["accuracy"] <= hi
    d = paired_bootstrap_diff(lambda a, b: float((a == b).mean()), y, y, p, n_boot=300)
    assert d["diff"] > 0 and d["p_one_sided"] < 0.5
    assert mcnemar_exact(y, y, y)["p_value"] == 1.0
    # 10 discordant pairs all favouring A -> p = 2 * 0.5**10
    yt = np.zeros(10, int)
    assert mcnemar_exact(yt, yt, np.ones(10, int))["p_value"] == pytest.approx(2 * 0.5 ** 10)
