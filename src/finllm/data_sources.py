"""Download, clean and split every dataset the project uses.

Dataset                         Used for                 Licence
------------------------------  --------------  -------------------------------------------
Financial PhraseBank v1.0       sentiment, tokenizer     CC BY-NC-SA 3.0 (Malo et al., 2014)
  (takala/financial_phrasebank)
Financial news headlines        pretraining, backtest    not stated on the HF card; mirror of the Kaggle
  (ashraq/financial-news)                       "Massive Stock News Analysis DB" raw_partner_headlines
finance-alpaca                  instruction tuning       MIT (Alpaca + FiQA + GPT-3.5 pairs)
  (gbharti/finance-alpaca)
The Verdict (E. Wharton, 1908)  general-English control  public domain (kept in data/samples)

Nothing downloaded is committed to git (see .gitignore); each user downloads
from the original source.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .gpt2_weights import download_file
from .paths import PROCESSED_DIR, RAW_DIR, SAMPLES_DIR

PHRASEBANK_URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/"
                  "data/FinancialPhraseBank-v1.0.zip")
NEWS_URL = ("https://huggingface.co/datasets/ashraq/financial-news/resolve/main/data/"
            "train-00000-of-00001-8ec327f23bbe0948.parquet")
FINANCE_ALPACA_URL = "https://huggingface.co/datasets/gbharti/finance-alpaca/resolve/main/Cleaned_date.json"

MIN_NEWS_DATE = pd.Timestamp("2000-01-01")
NEWS_CACHE_VERSION = "v2"   # bump whenever clean_news changes so stale caches are ignored

LABELS = ["negative", "neutral", "positive"]          # label_id 0, 1, 2
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
# File name in the zip -> minimum share of the 16 annotators who agreed.
PHRASEBANK_FILES = {"Sentences_50Agree.txt": 50, "Sentences_66Agree.txt": 66,
                    "Sentences_75Agree.txt": 75, "Sentences_AllAgree.txt": 100}


# ---------------------------------------------------------------------------
# The Verdict (general-English control text)
# ---------------------------------------------------------------------------

def load_the_verdict() -> str:
    return (SAMPLES_DIR / "the-verdict.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Financial PhraseBank
# ---------------------------------------------------------------------------

def parse_phrasebank_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Parse the four agreement files into one de-duplicated table.

    Columns: text, label, label_id, agreement (highest agreement level the
    sentence appears in: 50 / 66 / 75 / 100). Sentences that appear with
    conflicting labels are dropped -- a small but real source of label noise.
    """
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for fname, level in PHRASEBANK_FILES.items():
            if fname not in names:
                continue
            raw = zf.read(names[fname]).decode("latin-1")
            for line in raw.splitlines():
                line = line.strip()
                if "@" not in line:
                    continue
                text, label = line.rsplit("@", 1)
                label = label.strip().lower()
                if label in LABEL2ID and text.strip():
                    rows.append((" ".join(text.split()), label, level))
    df = pd.DataFrame(rows, columns=["text", "label", "agreement"])
    if df.empty:
        raise ValueError("No sentences parsed from the PhraseBank zip")
    n_labels = df.groupby("text")["label"].nunique()
    conflicting = set(n_labels[n_labels > 1].index)
    df = df[~df["text"].isin(conflicting)]
    df = (df.groupby(["text", "label"], as_index=False)["agreement"].max()
            .sort_values("text").reset_index(drop=True))
    df["label_id"] = df["label"].map(LABEL2ID)
    df.attrs["n_conflicting_dropped"] = len(conflicting)
    return df[["text", "label", "label_id", "agreement"]]


def load_phrasebank(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = download_file(PHRASEBANK_URL, Path(raw_dir) / "FinancialPhraseBank-v1.0.zip")
    return parse_phrasebank_zip(path.read_bytes())


def split_phrasebank(df: pd.DataFrame, seed: int = 123, val_frac: float = 0.1,
                     test_frac: float = 0.2) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split (the same split is reused by the classifier and the instruction-tuned model)."""
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(df, test_size=test_frac, stratify=df["label_id"], random_state=seed)
    rel_val = val_frac / (1 - test_frac)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label_id"],
                                  random_state=seed)
    return {k: v.reset_index(drop=True) for k, v in
            {"train": train, "val": val, "test": test}.items()}


def phrasebank_splits(seed: int = 123, processed_dir: Path = PROCESSED_DIR,
                      raw_dir: Path = RAW_DIR, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Load the persisted split, creating it on first use so every experiment sees the same test set."""
    processed_dir = Path(processed_dir)
    files = {k: processed_dir / f"phrasebank_{k}.csv" for k in ("train", "val", "test")}
    if all(p.exists() for p in files.values()) and df is None:
        return {k: pd.read_csv(p) for k, p in files.items()}
    df = load_phrasebank(raw_dir) if df is None else df
    splits = split_phrasebank(df, seed=seed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for k, p in files.items():
        splits[k].to_csv(p, index=False)
    return splits


# ---------------------------------------------------------------------------
# Dated, ticker-tagged news headlines
# ---------------------------------------------------------------------------

def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns to date (midnight, tz-naive), ticker, headline, publisher;
    drop blanks and exact duplicates of (ticker, date, headline)."""
    df = df.rename(columns={"stock": "ticker", "title": "headline"})
    df = df[[c for c in ["date", "ticker", "headline", "publisher"] if c in df.columns]].copy()
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = dates.dt.tz_convert(None).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["headline"] = df["headline"].astype(str).str.split().str.join(" ")
    df = df.dropna(subset=["date"])
    # Placeholder timestamps (e.g. 1970-01-01 / 1969-12-31 = "time zero") and future dates are
    # data errors; one such row would stretch the calendar windows by decades.
    bad_date = (df["date"] < MIN_NEWS_DATE) | (df["date"] > pd.Timestamp.now().normalize())
    n_bad = int(bad_date.sum())
    df = df[~bad_date]
    df = df[(df["headline"].str.len() > 3) & (df["ticker"].str.len() > 0)]
    key = df["headline"].str.lower()
    df = df.loc[~pd.DataFrame({"t": df["ticker"], "d": df["date"], "h": key}).duplicated()]
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df.attrs["n_bad_dates_dropped"] = n_bad
    return df


def load_news(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = download_file(NEWS_URL, Path(raw_dir) / "financial-news.parquet")
    return clean_news(pd.read_parquet(path))


def load_news_cached(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Cleaned news, cached as parquet after the first (slower) clean."""
    cache = Path(processed_dir) / f"news_clean_{NEWS_CACHE_VERSION}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_news(raw_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def calendar_periods(dates: pd.Series, pretrain_frac: float = 0.4, is_frac: float = 0.3) -> dict:
    """Split the news calendar (not the row count) into three consecutive windows:

        [start, pretrain_end]   domain pretraining text (never traded)
        (pretrain_end, is_end]  in-sample backtest: all design choices made here
        (is_end, end]           out-of-sample backtest: run once, reported as is
    """
    start, end = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    span = end - start
    pretrain_end = (start + span * pretrain_frac).normalize()
    is_end = (start + span * (pretrain_frac + is_frac)).normalize()
    # Guard against outlier dates silently emptying a window.
    n = len(dates)
    counts = {"pretrain": int((dates <= pretrain_end).sum()),
              "in_sample": int(((dates > pretrain_end) & (dates <= is_end)).sum()),
              "out_of_sample": int((dates > is_end).sum())}
    if n >= 1000 and min(counts.values()) < 0.01 * n:
        raise ValueError(f"A calendar window holds <1% of headlines {counts} for dates "
                         f"{start.date()}..{end.date()} - check for outlier dates in the news file")
    return {"start": start, "pretrain_end": pretrain_end, "is_end": is_end, "end": end}


def headlines_corpus(news: pd.DataFrame, end_date, max_chars: int, seed: int = 123,
                     start_date=None) -> list[str]:
    """Random sample of unique headlines dated <= end_date, capped at ~max_chars."""
    mask = news["date"] <= pd.Timestamp(end_date)
    if start_date is not None:
        mask &= news["date"] > pd.Timestamp(start_date)
    heads = news.loc[mask, "headline"].drop_duplicates()
    heads = heads.sample(frac=1.0, random_state=seed)
    lengths = heads.str.len().to_numpy() + 1
    keep = int(np.searchsorted(np.cumsum(lengths), max_chars)) + 1
    return heads.iloc[:keep].tolist()


# ---------------------------------------------------------------------------
# finance-alpaca (instruction data)
# ---------------------------------------------------------------------------

def load_finance_alpaca(raw_dir: Path = RAW_DIR) -> list[dict]:
    path = download_file(FINANCE_ALPACA_URL, Path(raw_dir) / "finance-alpaca.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for d in data:
        instr = str(d.get("instruction", "") or "").strip()
        resp = str(d.get("output", "") or "").strip()
        if instr and resp:
            out.append({"instruction": instr, "input": str(d.get("input", "") or "").strip(),
                        "output": resp})
    return out
