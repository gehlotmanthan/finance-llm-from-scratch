"""Statistics used to judge results honestly.

Classification (sentiment models)
    classification_metrics       accuracy, macro-F1, per-class P/R/F1, confusion matrix
    bootstrap_ci                 percentile bootstrap CI for any metric on a test set
    paired_bootstrap_diff        CI and p-value for "model A beats model B" on the same test set
    mcnemar_exact                exact McNemar test on disagreeing predictions

Trading (backtest)
    perf_summary                 annualised return / vol / Sharpe, max drawdown, hit rate
    newey_west_tstat             t-stat of a mean robust to autocorrelation & heteroskedasticity
    block_bootstrap_sharpe_ci    circular block bootstrap CI for the Sharpe ratio
"""

import math

import numpy as np

# ----------------------------------------------------------------------------- classification

def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[t, p] += 1
    return cm


def macro_f1(y_true, y_pred, n_classes: int = 3) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    f1s = []
    for k in range(n_classes):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
        rec = tp / cm[k, :].sum() if cm[k, :].sum() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    k = len(labels)
    cm = confusion_matrix(y_true, y_pred, k)
    per = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        per[lab] = {"precision": float(prec), "recall": float(rec),
                    "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                    "support": int(cm[i, :].sum())}
    return {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred, k),
            "per_class": per, "confusion_matrix": cm.tolist(), "n": int(len(y_true))}


def bootstrap_ci(metric, y_true, y_pred, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap: resample test rows with replacement, recompute the metric."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [metric(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap_diff(metric, y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """metric(A) - metric(B) on the same resampled rows. p = share of resamples where A <= B."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx]))
    diffs = np.array(diffs)
    return {"diff": float(metric(y_true, pred_a) - metric(y_true, pred_b)),
            "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "p_one_sided": float(np.mean(diffs <= 0))}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact two-sided McNemar test: do A and B have different error rates?"""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_right_b_wrong = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    a_wrong_b_right = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    n = a_right_b_wrong + a_wrong_b_right
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    k = min(a_right_b_wrong, a_wrong_b_right)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"a_right_b_wrong": a_right_b_wrong, "a_wrong_b_right": a_wrong_b_right,
            "p_value": float(min(1.0, 2 * p))}


# ----------------------------------------------------------------------------- trading

def max_drawdown(returns) -> float:
    """Largest peak-to-trough fall of the cumulative (compounded) equity curve, as a positive number."""
    eq = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(np.max(1 - eq / peak)) if len(eq) else 0.0


def newey_west_tstat(x, lags: int | None = None) -> float:
    """t-stat of mean(x) with Newey-West (Bartlett kernel) standard errors."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    s = e @ e / n
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        s += 2 * w * (e[l:] @ e[:-l]) / n
    se = math.sqrt(max(s, 1e-300) / n)
    return float(x.mean() / se)


def sharpe(returns, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")


def perf_summary(returns, periods_per_year: int = 252) -> dict:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return {"n_days": 0}
    ann_ret = float((1 + r).prod() ** (periods_per_year / len(r)) - 1)
    ann_vol = float(r.std(ddof=1) * math.sqrt(periods_per_year)) if len(r) > 1 else float("nan")
    mdd = max_drawdown(r)
    return {"n_days": int(len(r)), "ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe(r, periods_per_year), "max_drawdown": mdd,
            "calmar": ann_ret / mdd if mdd > 0 else float("nan"),
            "hit_rate": float(np.mean(r[r != 0] > 0)) if np.any(r != 0) else float("nan"),
            "mean_daily_bps": float(r.mean() * 1e4),
            "nw_tstat": newey_west_tstat(r)}


def block_bootstrap_sharpe_ci(returns, block: int = 10, n_boot: int = 2000, seed: int = 0,
                              periods_per_year: int = 252) -> tuple[float, float]:
    """Circular block bootstrap keeps short-range autocorrelation inside each block."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < block * 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        out.append(sharpe(r[idx], periods_per_year))
    return float(np.nanquantile(out, 0.025)), float(np.nanquantile(out, 0.975))
