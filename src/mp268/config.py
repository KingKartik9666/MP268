"""Shared helpers: find the repo root, read params.yaml, hash the original data file."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]  # src/mp268/config.py -> repo root


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    """Read params.yaml with yaml.safe_load (never yaml.load: it can run code)."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str | Path) -> Path:
    """Turn a path from params.yaml (relative to the repo root) into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else ROOT / p


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's raw bytes. Logged as `data_sha256` in every MLflow run."""
    with open(resolve(path), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a data file exactly as given (dates stay 'YYYY-MM-DD' strings). Never writes back."""
    return pd.read_csv(resolve(path))
