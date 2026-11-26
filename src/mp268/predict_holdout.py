"""Final prediction (PROBLEM.md section 8): the @champion model predicts every holdout row.
Owner: Kartik.

Run from the repo root after `python -m mp268.monitor`:
    python -m mp268.predict_holdout
Writes predictions/holdout_predictions.csv with exactly two columns: record_id,prediction
"""
from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from mp268.clean import clean_dataframe
from mp268.config import load_params, read_raw, resolve
from mp268.train import feature_columns, setup_mlflow


def predict_holdout(params: dict) -> pd.DataFrame:
    setup_mlflow(params)
    name = params["mlflow"]["registered_model_name"]
    champion = MlflowClient().get_model_version_by_alias(name, "champion")
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")

    raw = read_raw(params["paths"]["holdout"])
    holdout = clean_dataframe(raw, params, is_train=False)   # same category spellings as training
    out = pd.DataFrame({"record_id": raw[params["columns"]["id"]],
                        "prediction": model.predict(holdout[feature_columns(params)]).round(1)})

    # the marking tool needs every record exactly once and a number for each
    assert len(out) == len(raw) and out["record_id"].is_unique, "one row per holdout record"
    assert out["prediction"].notna().all(), "no blank predictions"

    path = resolve(params["paths"]["predictions_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"champion = version {champion.version}; wrote {len(out)} predictions -> {path}")
    return out


if __name__ == "__main__":
    predict_holdout(load_params())
