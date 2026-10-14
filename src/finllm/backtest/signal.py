"""Turning per-headline sentiment into a per-stock, per-day signal without look-ahead.

Timing convention (the conservative choice for date-only timestamps):

    headline dated day d (time unknown -- could be after the close)
      -> first trading day strictly AFTER d = trade day t
      -> enter at the OPEN of t, exit at the open of t+h

Weekend / holiday news therefore rolls into the next session, and nothing
published on day d is ever traded on day d.

Score per headline = P(positive) - P(negative), in [-1, 1]. A stock's score on
trade day t is the mean over all its headlines that map to t.
"""

import numpy as np
import pandas as pd


def sentiment_score(probs: np.ndarray) -> np.ndarray:
    """probs columns ordered (negative, neutral, positive)."""
    probs = np.asarray(probs)
    return probs[:, 2] - probs[:, 0]


def next_trading_day(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    cal = np.asarray(calendar.values)
    pos = np.searchsorted(cal, np.asarray(pd.to_datetime(dates).values), side="right")
    out = np.full(len(pos), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = pos < len(cal)
    out[ok] = cal[pos[ok]]
    return pd.Series(out, index=dates.index)


def build_signal(scored: pd.DataFrame, calendar: pd.DatetimeIndex, score_col: str) -> pd.DataFrame:
    """scored: one row per headline with date, ticker and ``score_col``.
    Returns one row per (trade_date, ticker) with mean score and headline count."""
    df = scored[["date", "ticker", score_col]].copy()
    df["trade_date"] = next_trading_day(df["date"], calendar)
    df = df.dropna(subset=["trade_date"])
    g = df.groupby(["trade_date", "ticker"])[score_col]
    return pd.DataFrame({"score": g.mean(), "n_news": g.size()}).reset_index()


def to_wide(signal: pd.DataFrame, value: str = "score") -> pd.DataFrame:
    return signal.pivot(index="trade_date", columns="ticker", values=value).sort_index()
