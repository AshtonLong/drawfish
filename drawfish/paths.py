"""Filesystem locations.

Code and the final model live in the repo. Heavy artifacts (virtualenv, Stockfish,
training shards, checkpoints, benchmark logs) live in DRAWFISH_HOME, which defaults
to %LOCALAPPDATA%\\Drawfish so they stay out of OneDrive sync.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "models"
DEFAULT_MODEL = MODELS_DIR / "drawfish.pt"

HOME = Path(os.environ.get("DRAWFISH_HOME") or
            Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Drawfish")
DATA_DIR = HOME / "data"
CKPT_DIR = HOME / "checkpoints"
LOG_DIR = HOME / "logs"


def stockfish_path() -> Path:
    """Locate a Stockfish binary (only needed for data generation and benchmarks)."""
    env = os.environ.get("STOCKFISH")
    if env:
        return Path(env)
    for p in sorted((HOME / "stockfish_dl").rglob("stockfish*.exe")):
        return p
    for p in sorted((HOME / "stockfish_dl").rglob("stockfish*")):
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Stockfish not found. Set the STOCKFISH env var or unpack it under "
        f"{HOME / 'stockfish_dl'}")
