"""Run all seven data-problem checks on the raw train.csv and print the `data_issues`
block of findings.yaml (handbook section 7).

Owner: Kartik. Kartik's two checks (leaking_column, target_typos) live here. The other five
are the teammates' functions, imported from wherever they currently are and translated into
the exact findings.yaml keys. A check that is missing or returns the wrong shape is printed as
a TODO instead of a guess, so nothing in findings.yaml is ever invented.

Run from the repo root (the root must be the current folder, so teammates' files are found):
    pip install -e .            # once, makes the `mp268` package importable
    python -m mp268.findings
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from mp268.config import load_params, read_raw


# ------------------------------------------------------------------ Kartik's checks
def leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str,
                   min_abs_corr: float = 0.95) -> dict:
    """A column filled in only after the day is over, so it secretly contains the answer.

    Both handbook clues must hold:
      1. the column is in train.csv but missing from holdout_features.csv
         (so it is not known at prediction time), and
      2. it agrees almost exactly with the target: |Spearman rank corr| >= min_abs_corr.
    Rank correlation is used so a few typos or sentinel values cannot hide the match.
    """
    best_col, best_corr = None, 0.0
    for col in train.columns:
        if col == target or col in holdout.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        corr = abs(train[col].corr(train[target], method="spearman"))
        if corr >= min_abs_corr and corr > best_corr:
            best_col, best_corr = col, corr
    if best_col is None:
        return {"present": False, "column": None}
    return {"present": True, "column": best_col}


def target_typos(train: pd.DataFrame, target: str, min_gap_ratio: float = 1.5) -> dict:
    """A few targets typed with an extra zero (10 times too big).

    Sort the target. Normal values rise smoothly, so neighbouring sorted values differ by a
    ratio close to 1. A typo cluster sits far above everything else with an empty gap below
    it, so the biggest neighbour ratio in the upper half jumps to several times.
    count = number of rows above that gap (rows as given, duplicates included).
    """
    y = np.sort(train[target].dropna().to_numpy(dtype=float))
    y = y[y > 0]
    upper = np.flatnonzero(y[:-1] >= np.median(y))   # only look for the gap above the median
    if len(upper) == 0:
        return {"present": False, "count": None}
    ratios = y[upper + 1] / y[upper]
    big = np.flatnonzero(ratios >= min_gap_ratio)
    if len(big) == 0:
        return {"present": False, "count": None}
    # the FIRST big gap is the one between normal values and the typos; later big gaps can
    # sit between two typos (e.g. 80k -> 250k) and must not split the typo cluster
    last_normal = y[upper[big[0]]]
    return {"present": True, "count": int((train[target] > last_normal).sum())}


# ------------------------------------------------------------------ teammates' checks
def _find(func_name: str, modules: list[str]):
    """Return the first `func_name` found in `modules` (lets teammates move their files)."""
    for mod in modules:
        try:
            fn = getattr(importlib.import_module(mod), func_name, None)
        except ImportError:
            continue
        if fn is not None:
            return fn
    return None


def _todo(why: str) -> str:
    return f"TODO: {why}"


def unit_mixup(train: pd.DataFrame, params: dict):
    fn = _find("check_unit_mixup", ["mp268.checks"])
    if fn is None:
        return _todo("Shubh: check_unit_mixup not found")
    r = fn(train, params["features"]["numeric"], params["columns"]["group"])
    if not r.get("present"):
        return {"present": False, "column": None, "group_value": None, "factor": None}
    d = r["details"]
    return {"present": True, "column": d["column"], "group_value": d["affected_group"],
            "factor": int(d["factor"])}


def stuck_sensor(train: pd.DataFrame, params: dict):
    fn = _find("check_stuck_sensor", ["mp268.checks"])
    if fn is None:
        return _todo("Shubh: check_stuck_sensor not found")
    r = fn(train, params["features"]["numeric"])
    if not r.get("present"):
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    return {"present": True, "column": r["column"],
            "start_date": str(r["start_date"])[:10], "end_date": str(r["end_date"])[:10]}


def sentinel_value(train: pd.DataFrame, params: dict):
    fn = _find("sentinel_value", ["mp268.checks", "checks"])
    if fn is None:
        return _todo("Tarun: sentinel_value not found")
    r = fn(train)                                    # {column: {value: count}}
    if not r:
        return {"present": False, "column": None, "value": None, "count": None}
    column, value, count = max(((c, v, n) for c, vs in r.items() for v, n in vs.items()),
                               key=lambda t: t[2])
    return {"present": True, "column": column, "value": int(value), "count": int(count)}


def duplicate_rows(train: pd.DataFrame, params: dict):
    fn = _find("duplicate_rows", ["mp268.data_checks", "mp268.checks", "src.data_checks"])
    if fn is None:
        return _todo("Anuj: duplicate_rows not found")
    n = int(fn(train))
    return {"present": n > 0, "count": n if n > 0 else None}


def spelling_variants(train: pd.DataFrame, params: dict):
    fn = _find("spelling_variants", ["mp268.data_checks", "mp268.checks", "src.data_checks"])
    if fn is None:
        return _todo("Anuj: spelling_variants not found")
    r = fn(train, params["features"]["categorical"])
    if isinstance(r, dict) and {"present", "column", "true_levels"} <= set(r):
        return {k: r[k] for k in ("present", "column", "true_levels")}
    # Anuj's function returns {column: value_counts}. Map every raw spelling to its real level
    # (strip + lower case, then Shubh's cleaning.ownership_map) and compare the level counts.
    normalize = _find("normalize_category", ["mp268.data_checks", "mp268.checks", "src.data_checks"]) \
        or (lambda v: v.strip().lower())
    aliases = params["cleaning"]["ownership_map"]
    best = None                                       # (levels removed, column, true levels)
    for column, counts in r.items():
        raw = [str(v) for v in counts.index]
        real = {aliases.get(normalize(v), normalize(v)) for v in raw}
        removed = len(set(raw)) - len(real)
        if removed > 0 and (best is None or removed > best[0]):
            best = (removed, str(column), len(real))
    if best is None:
        return {"present": False, "column": None, "true_levels": None}
    return {"present": True, "column": best[1], "true_levels": int(best[2])}


def run_all(params: dict | None = None) -> dict:
    """The data_issues block, in findings.yaml order, from the raw files as given."""
    params = params or load_params()
    p, target = params["paths"], params["features"]["target"]
    train, holdout = read_raw(p["raw_train"]), read_raw(p["holdout"])
    c = params["checks"]
    return {
        "leaking_column": leaking_column(train, holdout, target, c["leak_min_abs_corr"]),
        "unit_mixup": unit_mixup(train, params),
        "sentinel_value": sentinel_value(train, params),
        "duplicate_rows": duplicate_rows(train, params),
        "stuck_sensor": stuck_sensor(train, params),
        "target_typos": target_typos(train, target, c["typo_min_gap_ratio"]),
        "spelling_variants": spelling_variants(train, params),
    }


if __name__ == "__main__":
    import yaml

    print(yaml.safe_dump({"data_issues": run_all()}, sort_keys=False))
