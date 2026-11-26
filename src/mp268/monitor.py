"""Two months later: drift, per-zone error, retraining and the promotion rule (PROBLEM.md section 7).
Owner: Kartik.

Run from the repo root after `python -m mp268.train final`:
    python -m mp268.monitor

1. PSI (10 bins) for every numeric feature: cleaned train.csv (reference) vs later.csv
2. v1's MAE on later.csv for each zone -> the worst-served zone
3. later.csv sorted by date: v2 = same model settings trained on cleaned train + first 70% of later
4. v1 vs v2 on the last 30%; promote v2 (alias @champion) only if its MAE is at least 8% lower
"""
from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from mp268.clean import clean_dataframe
from mp268.config import file_sha256, load_params, read_raw, resolve
from mp268.train import (build_pipeline, export_champion, feature_columns, load_clean_train,
                         log_and_register, scores, set_champion, setup_mlflow)


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of `actual` against the reference `expected`.

    Bin edges are the deciles of the reference (10 bins with about 10% of the reference each);
    the outer edges are open so values outside the reference range land in the end bins.
    PSI = sum((a - e) * ln(a / e)) over bins, with empty bins set to eps (ln(0) is undefined).
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift.
    NOTE: check the binning against the course's P13 lab and change `bins`/edges here if it differs.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_clean_later(params: dict) -> pd.DataFrame:
    """later.csv is already clean; the same rules are applied anyway (no-ops) for consistency."""
    later = clean_dataframe(read_raw(params["paths"]["raw_later"]), params, is_train=False)
    date = params["columns"]["date"]
    return later.assign(**{date: pd.to_datetime(later[date])})


def mae_by_group(model, df: pd.DataFrame, feats: list[str], target: str, group: str) -> dict:
    err = (pd.Series(model.predict(df[feats]), index=df.index) - df[target]).abs()
    return {str(k): round(float(v), 1) for k, v in err.groupby(df[group]).mean().items()}


def run(params: dict, force: bool = False) -> dict:
    setup_mlflow(params)
    name = params["mlflow"]["registered_model_name"]
    versions = [int(v.version) for v in MlflowClient().search_model_versions(f"name='{name}'")]
    if 1 not in versions:
        raise SystemExit("no version 1 registered: run `python -m mp268.train final` first")
    if 2 in versions and not force:
        raise SystemExit(f"{name} already has version 2 (v2). Use --force to register another version.")

    target, feats = params["features"]["target"], feature_columns(params)
    group, date = params["columns"]["group"], params["columns"]["date"]
    train, later = load_clean_train(params), load_clean_later(params)

    # 1. drift: PSI per numeric feature (blank staff values from the 9999 sentinel are left out)
    psi_vals = {c: round(psi(train[c].dropna(), later[c].dropna()), 4) for c in params["features"]["numeric"]}
    most_drifted = max(psi_vals, key=psi_vals.get)

    # 2. v1 (registered version 1) on all of later.csv, per zone
    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")
    mae_zone = mae_by_group(v1, later, feats, target, group)
    worst_group = max(mae_zone, key=mae_zone.get)
    mae_later_v1 = round(scores(later[target], v1.predict(later[feats]))["mae"], 1)

    # 3. retrain v2 on cleaned train + the first 70% of later.csv (sorted by date)
    later_sorted = later.sort_values(date, kind="mergesort").reset_index(drop=True)
    n_first = int(len(later_sorted) * 0.7)
    first, check = later_sorted.iloc[:n_first], later_sorted.iloc[n_first:]
    v1_info = json.loads(resolve("reports/v1.json").read_text())
    cfg = next(m for m in params["models"] if m["name"] == v1_info["model"])
    v2_data = pd.concat([train, first], ignore_index=True)
    v2 = build_pipeline(cfg, params).fit(v2_data[feats], v2_data[target])

    # 4. compare on the last 30% and apply the client's rule
    mae_v1_check = round(scores(check[target], v1.predict(check[feats]))["mae"], 1)
    mae_v2_check = round(scores(check[target], v2.predict(check[feats]))["mae"], 1)
    improvement = (mae_v1_check - mae_v2_check) / mae_v1_check
    decision = "promote" if improvement >= 0.08 else "keep"

    with mlflow.start_run(run_name=f"v2_{cfg['name']}") as run:
        mlflow.set_tags({"stage": "retrain", "model_version_role": "v2", "decision": decision})
        mlflow.log_params({"data_sha256": file_sha256(params["paths"]["raw_train"]),
                           "later_sha256": file_sha256(params["paths"]["raw_later"]),
                           "model_name": cfg["name"], "model_type": cfg["type"],
                           "n_train_v2": len(v2_data), "n_later_first": len(first), "n_later_check": len(check),
                           "check_start": str(check[date].min().date()), "most_drifted_feature": most_drifted,
                           "worst_served_group": worst_group, "decision": decision})
        mlflow.log_metrics({**{f"psi_{c}": v for c, v in psi_vals.items()},
                            **{f"mae_later_v1_{g}": v for g, v in mae_zone.items()},
                            "mae_later_v1": mae_later_v1, "mae_v1_check": mae_v1_check,
                            "mae_v2_check": mae_v2_check, "mae_improvement_pct": round(100 * improvement, 2)})
        v2_version = log_and_register(v2, v2_data[feats], params)
        run_id = run.info.run_id

    if decision == "promote":
        set_champion(params, v2_version)
        export_champion(v2, params)
    champion = MlflowClient().get_model_version_by_alias(name, "champion").version

    out = {"psi": psi_vals, "most_drifted_feature": most_drifted, "mae_by_group": mae_zone,
           "worst_served_group": worst_group, "mae_later_v1": mae_later_v1,
           "mae_v1_on_check": mae_v1_check, "mae_v2_on_check": mae_v2_check,
           "improvement_pct": round(100 * improvement, 2), "decision": decision,
           "v2_run_id": run_id, "v2_version": int(v2_version), "champion_version": int(champion),
           "n_later_first": len(first), "n_later_check": len(check)}
    resolve("reports/monitoring.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="register another v2 even if version 2 exists")
    args = ap.parse_args(argv)
    print(json.dumps(run(load_params(), force=args.force), indent=2))


if __name__ == "__main__":
    main()
