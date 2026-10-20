"""Backtest: look-ahead-safe alignment, portfolio maths, trading statistics."""

import numpy as np
import pandas as pd
import pytest

from finllm import synthetic
from finllm.backtest.portfolio import (daily_ic, forward_open_returns, long_short_weights, permutation_test,
                                       run_book, tradable_mask)
from finllm.backtest.signal import build_signal, next_trading_day, sentiment_score, to_wide
from finllm.data_sources import calendar_periods, clean_news, headlines_corpus
from finllm.stats import max_drawdown, newey_west_tstat, perf_summary, sharpe

CAL = pd.bdate_range("2020-01-01", "2020-01-31")  # weekdays only


def test_next_trading_day_never_same_day():
    d = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-06", "2020-01-31"]))  # Fri, Sat, Mon, last
    out = next_trading_day(d, CAL)
    assert out.iloc[0] == pd.Timestamp("2020-01-06")   # Friday news -> Monday open
    assert out.iloc[1] == pd.Timestamp("2020-01-06")   # Saturday news -> Monday open
    assert out.iloc[2] == pd.Timestamp("2020-01-07")   # Monday news -> Tuesday open
    assert pd.isna(out.iloc[3])                        # no next day in calendar -> dropped


def test_build_signal_aggregates_weekend_news():
    scored = pd.DataFrame({"date": pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-04"]),
                           "ticker": ["A", "A", "B"], "s": [1.0, 0.0, -1.0]})
    sig = build_signal(scored, CAL, "s")
    a = sig[sig.ticker == "A"].iloc[0]
    assert a.trade_date == pd.Timestamp("2020-01-06") and a.score == 0.5 and a.n_news == 2
    assert to_wide(sig).loc["2020-01-06", "B"] == -1.0


def test_sentiment_score():
    assert sentiment_score(np.array([[0.7, 0.2, 0.1], [0.0, 0.0, 1.0]])).tolist() == pytest.approx([-0.6, 1.0])


def test_weights_are_dollar_neutral_and_respect_min_names():
    idx = pd.to_datetime(["2020-01-06", "2020-01-07"])
    score = pd.DataFrame({"A": [0.9, 0.9], "B": [0.8, np.nan], "C": [-0.9, -0.9], "D": [-0.5, 0.1]}, index=idx)
    fwd = pd.DataFrame(0.01, index=idx, columns=score.columns)
    trad = pd.DataFrame(True, index=idx, columns=score.columns)
    w = long_short_weights(score, fwd, trad, threshold=0.25, min_names=2)
    assert w.loc["2020-01-06"].tolist() == pytest.approx([0.25, 0.25, -0.25, -0.25])
    assert (w.loc["2020-01-07"] == 0).all()          # only 1 long and 1 short -> no trade
    assert w.sum(axis=1).abs().max() < 1e-12


def test_run_book_pnl_and_costs():
    idx = pd.to_datetime(["2020-01-06", "2020-01-07"])
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5]}, index=idx)
    fwd = pd.DataFrame({"A": [0.02, 0.0], "B": [-0.02, 0.0]}, index=idx)
    b = run_book(w, fwd, cost_bps=10)
    assert b["gross"].tolist() == pytest.approx([0.02, 0.0])
    assert b["turnover"].tolist() == pytest.approx([1.0, 0.0])
    assert b["net"].iloc[0] == pytest.approx(0.02 - 0.001)


def test_forward_returns_and_tradable_use_no_future_close():
    px = pd.DataFrame({"A": [10.0, 11.0, 12.1, 4.0]}, index=pd.bdate_range("2020-01-06", periods=4))
    r = forward_open_returns(px, 1)
    assert r["A"].iloc[0] == pytest.approx(0.1)
    m = tradable_mask(px, 5.0)
    assert not m["A"].iloc[0] and m["A"].iloc[1] and m["A"].iloc[3]   # uses yesterday's close


def _synthetic_backtest(lag_shift):
    news = synthetic.news_like(300, n_tickers=40, per_day=20, seed=1)
    opens, closes = synthetic.prices_like(news, seed=1, signal_bps=40)
    scores = news.assign(score=news["true_label"] - 1.0)
    if lag_shift:
        # Pretend the headline arrived one trading day LATER than it did: the signal is now
        # only knowable after the move it predicts, so a correct backtest must find nothing.
        cal = opens.index
        scores["date"] = next_trading_day(scores["date"], cal)
    sig = to_wide(build_signal(scores.dropna(subset=["date"]), opens.index, "score")).reindex(columns=opens.columns)
    fwd = forward_open_returns(opens, 1)
    w = long_short_weights(sig, fwd, tradable_mask(closes, 0.0), 0.25, 3)
    return run_book(w, fwd, 0.0)["gross"]


def test_backtest_finds_planted_signal_and_not_stale_signal():
    real = _synthetic_backtest(False)
    stale = _synthetic_backtest(True)
    assert newey_west_tstat(real.values) > 5
    assert abs(newey_west_tstat(stale.values)) < 3


def test_ic_and_permutation():
    idx = pd.bdate_range("2020-01-06", periods=60)
    rng = np.random.default_rng(0)
    s = pd.DataFrame(rng.normal(size=(60, 30)), index=idx)
    r = s * 0.01 + rng.normal(0, 0.01, (60, 30))
    ic = daily_ic(s, r)
    assert ic.mean() > 0.5
    trad = pd.DataFrame(True, index=idx, columns=s.columns)
    perm = permutation_test(s, r, trad, 0.5, 3, n_perm=50)
    assert perm["p_value_one_sided"] < 0.05


def test_trading_stats():
    assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(0.5)
    assert np.isnan(sharpe([0.01, 0.01, 0.01]))
    x = np.random.default_rng(0).normal(0.001, 0.01, 2000)
    t_classic = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert newey_west_tstat(x) == pytest.approx(t_classic, rel=0.15)
    s = perf_summary(x)
    assert s["n_days"] == 2000 and s["sharpe"] > 0


def test_clean_news_and_periods():
    raw = pd.DataFrame({"date": ["2015-01-02 00:00:00", "2015-01-02 00:00:00", "bad", "2016-06-01 00:00:00"],
                        "stock": ["aapl", "AAPL", "X", "MSFT"],
                        "headline": ["Apple  beats", "apple beats", "x x x x", "Microsoft falls"],
                        "publisher": ["p"] * 4})
    df = clean_news(raw)
    assert len(df) == 2 and df["ticker"].tolist() == ["AAPL", "MSFT"]
    per = calendar_periods(df["date"])
    assert per["start"] < per["pretrain_end"] < per["is_end"] < per["end"]
    assert headlines_corpus(df, per["pretrain_end"], 1000) == ["Apple beats"]


def test_download_prices_caches_and_retries_on_empty(tmp_path, monkeypatch):
    import sys
    import types
    from finllm.backtest.prices import download_prices, load_price_panel
    calls = {"n": 0}

    def fake_download(syms, **kw):
        calls["n"] += 1
        if calls["n"] == 1:                       # first call: rate-limited, nothing back
            return pd.DataFrame()
        idx = pd.bdate_range("2020-01-01", periods=30)
        frames = {s: pd.DataFrame({"Open": 10.0, "Close": 10.5, "Volume": 1e6}, index=idx)
                  for s in syms if s != "DEAD"}
        return pd.concat(frames, axis=1)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    log = lambda *a: None
    info = download_prices(["AAA", "BRK.B", "DEAD"], "2020-01-01", "2020-02-15", tmp_path, pause=0, log=log)
    assert info["missing"] == [] and not list(tmp_path.glob("*.parquet"))   # nothing marked dead after a blank
    info = download_prices(["AAA", "BRK.B", "DEAD"], "2020-01-01", "2020-02-15", tmp_path, pause=0, log=log)
    assert info["missing"] == ["DEAD"]
    panel = load_price_panel(["AAA", "BRK.B", "DEAD"], tmp_path)
    assert list(panel["open"].columns) == ["AAA", "BRK.B"] and len(panel["close"]) == 30


def test_clean_news_drops_placeholder_dates_and_periods_guard():
    raw = pd.DataFrame({"date": ["1969-12-31 00:00:00"] + ["2015-01-02 00:00:00"] * 3 + ["2018-01-02 00:00:00"],
                        "stock": ["A", "B", "C", "D", "E"], "headline": ["zero time", "b up", "c up", "d up", "e up"],
                        "publisher": ["p"] * 5})
    df = clean_news(raw)
    assert df["date"].min() == pd.Timestamp("2015-01-02") and df.attrs["n_bad_dates_dropped"] == 1
    lopsided = pd.Series([pd.Timestamp("1990-01-01")] + [pd.Timestamp("2018-01-01")] * 2000)
    with pytest.raises(ValueError):
        calendar_periods(lopsided)
