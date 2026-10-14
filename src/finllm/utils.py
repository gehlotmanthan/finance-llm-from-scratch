"""Small helpers shared by every chapter: seeding, device choice, result files."""

import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .paths import PROJECT_ROOT


def set_seed(seed: int = 123) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """cuda > mps > cpu, unless ``prefer`` names one explicitly."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{props.name} ({props.total_memory / 1024**3:.1f} GB)"
    return device.type


def bf16_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_bf16_supported()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def run_metadata(device: torch.device | None = None, **extra) -> dict:
    """Context saved next to every result so a number can always be traced back."""
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if device is not None:
        meta["device"] = device_name(device)
    meta.update(extra)
    return meta


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    return str(obj)


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """``with Timer() as t: ...`` then ``t.seconds``."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False
