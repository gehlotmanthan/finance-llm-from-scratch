"""Project folder layout. Every script reads and writes through these paths."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Downloaded data can live elsewhere (e.g. a bigger drive): set FINLLM_DATA_DIR.
DATA_DIR = Path(os.environ.get("FINLLM_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"              # untouched downloads
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned splits, cached scores
PRICES_DIR = DATA_DIR / "prices"        # cached daily OHLC per ticker
MODELS_DIR = DATA_DIR / "models"        # OpenAI GPT-2 safetensors
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"  # small public-domain files kept in git

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SMOKE_DIR = RESULTS_DIR / "smoke"


def results_dirs(smoke: bool):
    """Return (results_dir, figures_dir). Smoke runs never overwrite real results."""
    if smoke:
        out = SMOKE_DIR
        figs = SMOKE_DIR / "figures"
    else:
        out, figs = RESULTS_DIR, FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    return out, figs


def checkpoint_path(name: str, smoke: bool = False) -> Path:
    folder = CHECKPOINTS_DIR / "smoke" if smoke else CHECKPOINTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name
