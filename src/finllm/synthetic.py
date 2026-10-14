"""Synthetic stand-ins for every dataset, used ONLY by unit tests and ``--smoke`` runs.

They let the whole pipeline run on a CPU in seconds without downloads.
Numbers produced from them are meaningless and are written to results/smoke/,
never to results/.
"""

import numpy as np
import pandas as pd

_POS = ["beats estimates", "raises guidance", "record revenue", "upgraded to buy",
        "profit rises", "wins major contract", "dividend increased"]
_NEG = ["misses estimates", "cuts guidance", "revenue falls", "downgraded to sell",
        "posts loss", "faces probe", "dividend suspended"]
_NEU = ["to present at conference", "announces board meeting date", "files quarterly report",
        "names new CFO", "schedules earnings call", "completes share registration"]
_COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella plc", "Stark Industries",
              "Wayne Enterprises", "Hooli", "Soylent Oyj", "Vandelay AB", "Tyrell Inc"]


def phrasebank_like(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lab = rng.choice(3, p=[0.15, 0.55, 0.30])
        phrase = rng.choice([_NEG, _NEU, _POS][lab])
        comp = rng.choice(_COMPANIES)
        q = rng.integers(1, 5)
        text = f"{comp} {phrase} in Q{q} , EUR {rng.integers(1, 900)} mn #{i}"
        rows.append((text, ["negative", "neutral", "positive"][lab], int(lab),
                     int(rng.choice([50, 66, 75, 100]))))
    return pd.DataFrame(rows, columns=["text", "label", "label_id", "agreement"])


def news_like(n_days: int = 400, n_tickers: int = 30, per_day: int = 12, seed: int = 0,
              start: str = "2014-01-01") -> pd.DataFrame:
    """Headlines whose sentiment weakly predicts the *next* day's return (see prices_like)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for d in days:
        for _ in range(per_day):
            t = rng.choice(tickers)
            lab = rng.choice(3, p=[0.25, 0.45, 0.30])
            phrase = rng.choice([_NEG, _NEU, _POS][lab])
            rows.append((d, t, f"{t} {phrase}", "synthetic", int(lab)))
    return pd.DataFrame(rows, columns=["date", "ticker", "headline", "publisher", "true_label"])


def prices_like(news: pd.DataFrame, seed: int = 0, signal_bps: float = 30.0):
    """Wide open/close price frames where next-day open-to-open return = noise + signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(news["date"].min(), news["date"].max() + pd.Timedelta(days=10))
    tickers = sorted(news["ticker"].unique())
    rets = pd.DataFrame(rng.normal(0, 0.015, (len(dates), len(tickers))), index=dates, columns=tickers)
    lab = news.groupby(["date", "ticker"])["true_label"].mean() - 1.0   # in [-1, 1]
    for (d, t), s in lab.items():
        nxt = dates[dates > d]
        if len(nxt) >= 2:
            # news on day d -> enter at open of nxt[0] -> earn return to open of nxt[1]
            rets.loc[nxt[1], t] += s * signal_bps / 1e4
    opens = 50 * np.exp(rets.cumsum())
    closes = opens * np.exp(rng.normal(0, 0.005, opens.shape))
    return opens, closes


def instruction_like(n: int = 60, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    qs = [("What is a dividend?", "A dividend is a cash payment a company makes to its shareholders."),
          ("Define EBITDA.", "Earnings before interest, taxes, depreciation and amortisation."),
          ("What does a bond yield measure?", "The annual return an investor earns on a bond at its price."),
          ("Explain short selling.", "Selling borrowed shares to buy them back later at a lower price.")]
    out = []
    for i in range(n):
        q, a = qs[rng.integers(len(qs))]
        out.append({"instruction": q, "input": "", "output": a})
    return out
