"""Shared setup for the chapter scripts: import path, CLI flags, plot style."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (fixed order) + recessive ink for text/grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true",
                   help="tiny synthetic run on CPU to check the pipeline (results go to results/smoke/)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: best available)")
    return p


def setup_plots():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 2.0, "font.size": 10, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def savefig(fig, figs_dir: Path, name: str) -> Path:
    path = Path(figs_dir) / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return path


def rel(path: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_base_gpt2(model_name: str, device, smoke: bool, **cfg_overrides):
    """OpenAI GPT-2 weights in our GPTModel, or a tiny random model for --smoke."""
    import torch
    from finllm.config import TINY_CONFIG
    from finllm.gpt2_weights import load_pretrained_gpt2
    from finllm.model import GPTModel
    if smoke:
        torch.manual_seed(0)
        cfg = dict(TINY_CONFIG, **{k: v for k, v in cfg_overrides.items() if k != "context_length"})
        return GPTModel(cfg).to(device).eval(), cfg
    return load_pretrained_gpt2(model_name, device=device, **cfg_overrides)


def news_and_periods(smoke: bool):
    """Cleaned news + the three calendar windows (pretrain / in-sample / out-of-sample)."""
    from finllm import data_sources as ds
    from finllm import synthetic
    news = synthetic.news_like(400) if smoke else ds.load_news_cached()
    return news, ds.calendar_periods(news["date"])


setup_plots()
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
