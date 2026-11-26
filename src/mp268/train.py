"""Train and choose the v1 model with MLflow tracking (handbook section 8). Owner: Kartik.

Run from the repo root after `pip install -e .`:
    python -m mp268.train select   # one MLflow run per candidate in params.yaml; v1 = lowest mae_val
    python -m mp268.train final    # chosen model: test MAE once (fit on train+val), then refit on all
                                   # cleaned train, register as version 1, alias @champion, export joblib

Every run logs the parameter `data_sha256` (SHA-256 of the original data/raw/train.csv) and
metrics whose names contain `mae`. Tracking goes to mlflow.db in the repo root.
"""
from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mp268.clean import clean_dataframe
from mp268.config import ROOT, file_sha256, load_params, read_raw, resolve

MODEL_TYPES = {
    "dummy": DummyRegressor,
    "ridge": Ridge,
    "gbr": GradientBoostingRegressor,
    "rf": RandomForestRegressor,
}


# ------------------------------------------------------------------ data
def feature_columns(params: dict) -> list[str]:
    """The 8 fields of the API request, in a fixed order."""
    return params["features"]["numeric"] + params["features"]["categorical"]


def load_clean_train(params: dict) -> pd.DataFrame:
    """Raw train.csv -> Shubh's cleaning rules -> real datetime column."""
    raw = read_raw(params["paths"]["raw_train"])
    clean = clean_dataframe(raw, params, is_train=True)
    date = params["columns"]["date"]
    return clean.assign(**{date: pd.to_datetime(clean[date])})


def chrono_split(df: pd.DataFrame, date_col: str, val_size: float, test_size: float):
    """Oldest days -> train, next days -> validation, newest days -> test.

    A random split would let the model see the future (later days) while being scored on the
    past. Stable sort (mergesort) keeps rows of the same day in file order, so it is repeatable.
    """
    n = len(df)
    n_test, n_val = int(round(n * test_size)), int(round(n * val_size))
    n_train = n - n_val - n_test
    order = df.sort_values(date_col, kind="mergesort").index.to_numpy()
    parts = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
    return tuple(df.loc[np.sort(idx)].reset_index(drop=True) for idx in parts)


# ------------------------------------------------------------------ model
def build_pipeline(model_cfg: dict, params: dict) -> Pipeline:
    """Preprocessing + model in ONE object, so training and the API can never disagree.

    numeric: fill blanks (staff 9999 -> blank) with the TRAINING median, then scale
    categorical: one 0/1 column per level; an unseen level becomes all zeros instead of an error
    """
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), params["features"]["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), params["features"]["categorical"]),
    ])
    model = MODEL_TYPES[model_cfg["type"]](**model_cfg.get("params", {}))
    return Pipeline([("pre", pre), ("model", model)])


def scores(y_true, y_pred) -> dict:
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


# ------------------------------------------------------------------ MLflow
def setup_mlflow(params: dict) -> str:
    """Point MLflow at mlflow.db in the repo root and select the experiment."""
    uri = params["mlflow"]["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri[len(prefix):].startswith(("/", "\\")) and ":" not in uri[len(prefix):]:
        uri = prefix + (ROOT / uri[len(prefix):]).as_posix()        # relative -> repo root
    mlflow.set_tracking_uri(uri)
    name = params["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(name)
    return uri


def base_params(params: dict, sha: str) -> dict:
    return {"data_sha256": sha, "features": ",".join(feature_columns(params)),
            "target": params["features"]["target"]}


# ------------------------------------------------------------------ stages
def select(params: dict) -> dict:
    """One MLflow run per candidate: fit on train, score on validation. Returns the ranking."""
    setup_mlflow(params)
    sha = file_sha256(params["paths"]["raw_train"])
    target, feats = params["features"]["target"], feature_columns(params)
    df = load_clean_train(params)
    train, val, _ = chrono_split(df, params["columns"]["date"],
                                 params["split"]["val_size"], params["split"]["test_size"])

    results = []
    for cfg in params["models"]:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.set_tag("stage", "selection")
            mlflow.log_params({**base_params(params, sha), "model_name": cfg["name"],
                               "model_type": cfg["type"], "n_train": len(train), "n_val": len(val),
                               "train_end": str(train["date"].max().date()),
                               "val_end": str(val["date"].max().date()),
                               **{f"model.{k}": v for k, v in cfg.get("params", {}).items()}})
            pipe = build_pipeline(cfg, params).fit(train[feats], train[target])
            s_tr = scores(train[target], pipe.predict(train[feats]))
            s_va = scores(val[target], pipe.predict(val[feats]))
            mlflow.log_metrics({"mae_train": s_tr["mae"], "mae_val": s_va["mae"],
                                "rmse_val": s_va["rmse"]})
            results.append({"name": cfg["name"], "run_id": run.info.run_id,
                            "mae_train": round(s_tr["mae"], 1), "mae_val": round(s_va["mae"], 1),
                            "rmse_val": round(s_va["rmse"], 1)})

    results.sort(key=lambda r: r["mae_val"])
    out = {"data_sha256": sha, "chosen": results[0]["name"], "ranking": results}
    path = resolve("reports/model_selection.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def log_and_register(pipe: Pipeline, X_example: pd.DataFrame, params: dict) -> str:
    """Log the fitted pipeline to the active run and register it; returns the new version number."""
    info = mlflow.sklearn.log_model(
        sk_model=pipe, name="model", input_example=X_example.head(3),
        registered_model_name=params["mlflow"]["registered_model_name"],
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,  # handbook section 8
    )
    return str(info.registered_model_version)


def set_champion(params: dict, version: str) -> None:
    MlflowClient().set_registered_model_alias(params["mlflow"]["registered_model_name"], "champion", version)


def export_champion(pipe: Pipeline, params: dict) -> None:
    """The file the FastAPI service and the Docker image load (paths.model_path)."""
    path = resolve(params["paths"]["model_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def existing_versions(params: dict) -> list[str]:
    name = params["mlflow"]["registered_model_name"]
    try:
        return [v.version for v in MlflowClient().search_model_versions(f"name='{name}'")]
    except Exception:
        return []


def final(params: dict, force: bool = False) -> dict:
    """Build v1 from the model chosen by `select` and make it the @champion.

    1. fit on train + validation, score once on the newest 20% of days (mae_test)
    2. refit on ALL cleaned train rows (more data for the real model)
    3. log that model in an MLflow run, register it (version 1), alias champion, export joblib
    """
    setup_mlflow(params)
    if existing_versions(params) and not force:
        raise SystemExit(f"{params['mlflow']['registered_model_name']} already has versions "
                         f"{existing_versions(params)}; v1 exists. Use --force to register another.")
    sel = json.loads(resolve("reports/model_selection.json").read_text())
    cfg = next(m for m in params["models"] if m["name"] == sel["chosen"])
    chosen = next(r for r in sel["ranking"] if r["name"] == sel["chosen"])
    sha = file_sha256(params["paths"]["raw_train"])
    target, feats = params["features"]["target"], feature_columns(params)
    df = load_clean_train(params)
    train, val, test = chrono_split(df, params["columns"]["date"],
                                    params["split"]["val_size"], params["split"]["test_size"])

    tv = pd.concat([train, val], ignore_index=True)
    test_scores = scores(test[target], build_pipeline(cfg, params).fit(tv[feats], tv[target]).predict(test[feats]))

    pipe = build_pipeline(cfg, params).fit(df[feats], df[target])
    with mlflow.start_run(run_name=f"v1_{cfg['name']}") as run:
        mlflow.set_tags({"stage": "final", "model_version_role": "v1"})
        mlflow.log_params({**base_params(params, sha), "model_name": cfg["name"], "model_type": cfg["type"],
                           "n_train": len(df), "train_end": str(df["date"].max().date()),
                           **{f"model.{k}": v for k, v in cfg.get("params", {}).items()}})
        mlflow.log_metrics({"mae_val": chosen["mae_val"], "mae_test": test_scores["mae"],
                            "rmse_test": test_scores["rmse"],
                            "mae_train_all": scores(df[target], pipe.predict(df[feats]))["mae"]})
        version = log_and_register(pipe, df[feats], params)
        run_id = run.info.run_id
    set_champion(params, version)
    export_champion(pipe, params)
    out = {"experiment_name": params["mlflow"]["experiment_name"], "best_run_id": run_id,
           "registered_model_name": params["mlflow"]["registered_model_name"],
           "champion_version": int(version), "model": cfg["name"],
           "mae_val": chosen["mae_val"], "mae_test": round(test_scores["mae"], 1)}
    resolve("reports/v1.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "final"])
    ap.add_argument("--force", action="store_true", help="final: register another version even if v1 exists")
    args = ap.parse_args(argv)
    params = load_params()
    if args.stage == "select":
        out = select(params)
        print(pd.DataFrame(out["ranking"]).to_string(index=False))
        print(f"\nv1 candidate (lowest mae_val): {out['chosen']}  -> reports/model_selection.json")
    else:
        out = final(params, force=args.force)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
