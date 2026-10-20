"""The web app's page components: real numbers in, escaped HTML out."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webapp import ui  # noqa: E402


def test_results_band_reads_the_committed_results():
    res = ui.load_results(ROOT / "results")
    if "f1" not in res:  # results not generated yet in this checkout
        return
    html = ui.band(res)
    assert f"{res['f1']:.3f}" in html
    assert f"{res['f1_baseline']:.3f}" in html
    assert "fl-band" in html


def test_missing_results_render_nothing(tmp_path):
    assert ui.load_results(tmp_path) == {}
    assert ui.band({}) == ""
    assert "fl-cal" not in ui.hero({})


def test_user_text_is_escaped():
    evil = '<img src=x onerror="alert(1)">'
    rows = [{"text": evil, "negative": 0.1, "neutral": 0.2, "positive": 0.7, "label": "positive", "score": 0.6}]
    for html in (ui.sentiment_table(rows, None, ["negative", "neutral", "positive"], "m"),
                 ui.answer(evil), ui.continuation(evil, evil)):
        assert "<img" not in html and "&lt;img" in html


def test_disagreement_is_flagged():
    rows = [{"text": "a", "negative": 0.1, "neutral": 0.2, "positive": 0.7, "label": "positive", "score": 0.6}]
    same = ui.sentiment_table(rows, [{"label": "positive", "score": 0.4}], ["negative", "neutral", "positive"], "m")
    diff = ui.sentiment_table(rows, [{"label": "neutral", "score": 0.0}], ["negative", "neutral", "positive"], "m")
    assert "models disagree" not in same and "models disagree" in diff


def test_attention_grid_masks_the_future():
    tokens = ["Tesla", " shares", " fall"]
    mat = np.tril(np.full((3, 3), 1 / 3))
    html = ui.attention_grid(tokens, mat, 0, "head 1")
    n = len(tokens)
    assert html.count('class="fl-att-cell is-masked"') == n * (n - 1) // 2 + 1  # +1 for the legend swatch
    assert html.count("fl-att-row") == n


def test_signed_uses_a_real_minus():
    assert ui.signed(-0.42) == "−0.42"
    assert ui.signed(0.36) == "+0.36"
