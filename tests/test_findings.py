"""Tests for Kartik's data checks. Tiny in-memory frames only: no CSV files needed (CI has none)."""
import numpy as np
import pandas as pd

from mp268.findings import leaking_column, target_typos


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(8000, 30000, n)
    return pd.DataFrame({
        "temp_c": rng.uniform(5, 45, n),
        "tank_dip": y + rng.integers(-50, 50, n),   # measured after the day: tracks the target
        "litres_per_day": y,
    })


def test_leaking_column_found_when_missing_from_holdout_and_matches_target():
    train = _frame()
    holdout = train.drop(columns=["tank_dip", "litres_per_day"])
    assert leaking_column(train, holdout, "litres_per_day") == {"present": True, "column": "tank_dip"}


def test_leaking_column_absent_when_column_is_in_holdout():
    train = _frame()
    holdout = train.drop(columns=["litres_per_day"])          # tank_dip known at prediction time
    assert leaking_column(train, holdout, "litres_per_day")["present"] is False


def test_target_typos_counts_rows_typed_ten_times_too_big():
    train = _frame()
    train.loc[[3, 7, 11], "litres_per_day"] *= 10
    assert target_typos(train, "litres_per_day") == {"present": True, "count": 3}


def test_target_typos_keeps_a_spread_out_typo_cluster_together():
    train = _frame()
    train.loc[0, "litres_per_day"] = 8_000 * 10      # 80k
    train.loc[1, "litres_per_day"] = 29_000 * 10     # 290k: a big gap BETWEEN the two typos too
    assert target_typos(train, "litres_per_day") == {"present": True, "count": 2}


def test_target_typos_absent_on_smooth_target():
    assert target_typos(_frame(), "litres_per_day") == {"present": False, "count": None}
