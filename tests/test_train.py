"""Tests for the training helpers. Tiny in-memory data only (CI has no CSVs)."""
import numpy as np
import pandas as pd

from mp268.train import build_pipeline, chrono_split

PARAMS = {"features": {
    "numeric": ["highway_distance_km", "nearby_pumps", "price_rs", "trucks_passing", "temp_c", "staff"],
    "categorical": ["ownership", "zone"],
}}
FEATS = PARAMS["features"]["numeric"] + PARAMS["features"]["categorical"]


def _rows(n=40, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D")[rng.permutation(n)],
        "highway_distance_km": rng.uniform(0, 15, n), "nearby_pumps": rng.integers(0, 10, n),
        "price_rs": rng.uniform(90, 110, n), "trucks_passing": rng.integers(0, 2000, n),
        "temp_c": rng.uniform(5, 45, n), "staff": rng.integers(3, 20, n).astype(float),
        "ownership": rng.choice(["private", "public_sector", "cooperative"], n),
        "zone": rng.choice(["Highway", "City", "Industrial"], n),
        "litres_per_day": rng.uniform(5000, 35000, n),
    })


def test_chrono_split_keeps_time_order_and_no_overlap():
    train, val, test = chrono_split(_rows(), "date", val_size=0.2, test_size=0.2)
    assert (len(train), len(val), len(test)) == (24, 8, 8)
    assert train["date"].max() <= val["date"].min() and val["date"].max() <= test["date"].min()


def test_pipeline_handles_blank_staff_and_unseen_category():
    df = _rows()
    df.loc[:4, "staff"] = np.nan                                  # cleaned 9999 sentinel
    pipe = build_pipeline({"type": "ridge", "params": {"alpha": 1.0}}, PARAMS)
    pipe.fit(df[FEATS], df["litres_per_day"])
    new = df[FEATS].head(2).copy()
    new.loc[new.index[0], "zone"] = "Airport"                    # level never seen in training
    pred = pipe.predict(new)
    assert pred.shape == (2,) and np.isfinite(pred).all()
