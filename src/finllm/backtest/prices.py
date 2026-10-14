"""Daily prices: download once with yfinance, cache one parquet per ticker.

Survivorship bias warning: Yahoo only serves tickers that still trade (or
recently did). Companies that were delisted, merged or went bankrupt during
the sample drop out, which usually flatters a backtest. Coverage is reported
in the results; with CRSP access (e.g. via WRDS) pass ``--prices-dir`` to a
folder of per-ticker CSV/parquet files with Date, Open, Close columns.
"""

import time
from pathlib import Path

import pandas as pd

from ..paths import PRICES_DIR
from ..utils import load_json, save_json


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").replace("/", "-")


def _cache_file(prices_dir: Path, ticker: str) -> Path:
    return Path(prices_dir) / f"{ticker.replace('/', '_')}.parquet"


def download_prices(tickers, start, end, prices_dir: Path = PRICES_DIR, chunk: int = 50,
                    pause: float = 1.0, log=print) -> dict:
    """Fetch adjusted daily Open/Close/Volume for tickers not yet cached."""
    import yfinance as yf
    prices_dir = Path(prices_dir)
    prices_dir.mkdir(parents=True, exist_ok=True)
    missing_file = prices_dir / "_missing.json"
    known_missing = set(load_json(missing_file)) if missing_file.exists() else set()
    todo = [t for t in tickers if not _cache_file(prices_dir, t).exists() and t not in known_missing]
    log(f"prices: {len(tickers) - len(todo)} cached, {len(todo)} to download")
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        syms = {yahoo_symbol(t): t for t in batch}
        try:
            df = yf.download(list(syms), start=str(pd.Timestamp(start).date()),
                             end=str((pd.Timestamp(end) + pd.Timedelta(days=10)).date()),
                             auto_adjust=True, group_by="ticker", threads=True, progress=False)
        except Exception as e:  # network hiccup: skip this chunk, it will be retried next run
            log(f"  chunk {i // chunk}: {e}")
            continue
        for sym, t in syms.items():
            try:
                sub = df[sym] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub[["Open", "Close", "Volume"]].dropna(subset=["Open", "Close"])
            except KeyError:
                sub = pd.DataFrame()
            if len(sub) < 20:
                known_missing.add(t)
                continue
            sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
            sub.to_parquet(_cache_file(prices_dir, t))
        log(f"  downloaded {min(i + chunk, len(todo))}/{len(todo)}")
        save_json(sorted(known_missing), missing_file)
        time.sleep(pause)
    return {"requested": len(tickers), "missing": sorted(known_missing & set(tickers))}


def load_price_panel(tickers, prices_dir: Path = PRICES_DIR) -> dict[str, pd.DataFrame]:
    """Wide frames (date x ticker): open, close, volume. Accepts parquet or CSV per ticker."""
    opens, closes, vols = {}, {}, {}
    for t in tickers:
        f = _cache_file(prices_dir, t)
        if f.exists():
            df = pd.read_parquet(f)
        elif (Path(prices_dir) / f"{t}.csv").exists():
            df = pd.read_csv(Path(prices_dir) / f"{t}.csv", index_col=0, parse_dates=True)
        else:
            continue
        df.columns = [c.capitalize() for c in df.columns]
        opens[t], closes[t] = df["Open"], df["Close"]
        if "Volume" in df:
            vols[t] = df["Volume"]
    if not opens:
        raise FileNotFoundError(f"No price files found in {prices_dir}")
    return {"open": pd.DataFrame(opens).sort_index(), "close": pd.DataFrame(closes).sort_index(),
            "volume": pd.DataFrame(vols).sort_index()}
