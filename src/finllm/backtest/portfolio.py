"""Long/short portfolio construction, costs and signal diagnostics.

Book on trade day t (positions held from the open of t to the open of t+h):
  * candidates = stocks with a news signal on t, prior close >= min_price,
    and a valid forward return
  * long  stocks with score >= +threshold, equal weight, +0.5 gross in total
  * short stocks with score <= -threshold, equal weight, -0.5 gross in total
  * no trade on days where either side has fewer than ``min_names`` stocks
    (keeps the book dollar-neutral)
  * cost = cost_bps per unit of turnover, turnover = sum_i |w_i,t - w_i,t-1|

All thresholds and settings are fixed before looking at out-of-sample data.
"""

import numpy as np
import pandas as pd

from ..stats import newey_west_tstat


def forward_open_returns(open_px: pd.DataFrame, horizon: int = 1, clip: float = 0.4) -> pd.DataFrame:
    """r[t] = open[t+h] / open[t] - 1, clipped to +-clip to neutralise bad ticks."""
    r = open_px.shift(-horizon) / open_px - 1
    return r.clip(-clip, clip)


def same_day_close_returns(close_px: pd.DataFrame, clip: float = 0.4) -> pd.DataFrame:
    """close[d] / close[d-1] - 1: the price reaction ON the news day (not tradable, diagnostic only)."""
    return (close_px / close_px.shift(1) - 1).clip(-clip, clip)


def tradable_mask(close_px: pd.DataFrame, min_price: float = 5.0) -> pd.DataFrame:
    """Known before the open of t: yesterday's close was at least ``min_price``."""
    return close_px.shift(1) >= min_price


def long_short_weights(score: pd.DataFrame, fwd: pd.DataFrame, tradable: pd.DataFrame,
                       threshold: float = 0.25, min_names: int = 3) -> pd.DataFrame:
    score, fwd = score.align(fwd, join="left")
    tradable = tradable.reindex_like(score).fillna(False).astype(bool)
    valid = score.notna() & fwd.notna() & tradable
    longs = (score >= threshold) & valid
    shorts = (score <= -threshold) & valid
    nl, ns = longs.sum(axis=1), shorts.sum(axis=1)
    active = (nl >= min_names) & (ns >= min_names)
    w = longs.astype(float).div(nl.replace(0, np.nan), axis=0) * 0.5 \
        - shorts.astype(float).div(ns.replace(0, np.nan), axis=0) * 0.5
    w = w.where(active, 0.0, axis=0).fillna(0.0)
    return w


def run_book(weights: pd.DataFrame, fwd: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Daily gross P&L, turnover, cost and net P&L of a weight matrix."""
    fwd = fwd.reindex_like(weights).fillna(0.0)
    gross = (weights * fwd).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * cost_bps / 1e4
    long_leg = (weights.clip(lower=0) * fwd).sum(axis=1) * 2
    short_leg = (-(weights.clip(upper=0)) * fwd).sum(axis=1) * 2
    return pd.DataFrame({"gross": gross, "turnover": turnover, "cost": cost, "net": gross - cost,
                         "long_leg": long_leg, "short_leg": short_leg,
                         "n_long": (weights > 0).sum(axis=1), "n_short": (weights < 0).sum(axis=1)})


def daily_ic(score: pd.DataFrame, ret: pd.DataFrame, tradable: pd.DataFrame | None = None,
             min_names: int = 5) -> pd.Series:
    """Cross-sectional Spearman rank correlation between score and return, per day."""
    score, ret = score.align(ret, join="left")
    if tradable is not None:
        score = score.where(tradable.reindex_like(score).fillna(False).astype(bool))
    out = {}
    for d in score.index:
        s, r = score.loc[d], ret.loc[d]
        m = s.notna() & r.notna()
        if m.sum() >= min_names and s[m].nunique() > 1 and r[m].nunique() > 1:
            out[d] = s[m].rank().corr(r[m].rank())
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    return {"mean_ic": float(ic.mean()) if len(ic) else float("nan"),
            "ic_std": float(ic.std()) if len(ic) > 1 else float("nan"),
            "ic_ir_annual": float(ic.mean() / ic.std() * np.sqrt(252)) if len(ic) > 1 and ic.std() > 0 else float("nan"),
            "nw_tstat": newey_west_tstat(ic.values), "share_positive": float((ic > 0).mean()) if len(ic) else float("nan"),
            "n_days": int(len(ic))}


def permutation_test(score: pd.DataFrame, fwd: pd.DataFrame, tradable: pd.DataFrame, threshold: float,
                     min_names: int, n_perm: int = 500, seed: int = 0) -> dict:
    """Shuffle scores across stocks within each day (keeps which stocks had news and how many),
    rebuild the gross long/short book, and compare its mean return with the real one."""
    score, fwd = score.align(fwd, join="left")
    tr = tradable.reindex_like(score).fillna(False).astype(bool)
    valid = score.notna() & fwd.notna() & tr
    days = []
    for d in score.index:
        m = valid.loc[d].to_numpy()
        if m.sum() >= 2 * min_names:
            days.append((score.loc[d].to_numpy()[m], fwd.loc[d].to_numpy()[m]))

    def book_mean(pairs):
        rets = []
        for s, r in pairs:
            L, S = s >= threshold, s <= -threshold
            if L.sum() >= min_names and S.sum() >= min_names:
                rets.append(0.5 * r[L].mean() - 0.5 * r[S].mean())
            else:
                rets.append(0.0)
        return float(np.mean(rets)) if rets else float("nan")

    actual = book_mean(days)
    rng = np.random.default_rng(seed)
    null = np.array([book_mean([(rng.permutation(s), r) for s, r in days]) for _ in range(n_perm)])
    return {"actual_mean_daily_gross": actual, "null_mean": float(np.nanmean(null)),
            "null_std": float(np.nanstd(null)),
            "p_value_one_sided": float((np.sum(null >= actual) + 1) / (n_perm + 1)),
            "n_perm": n_perm, "null_sample": null[:500].round(6).tolist()}
