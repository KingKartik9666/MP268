import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

with open("params.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event_data"):
            payload.update(record.event_data)
        return json.dumps(payload)


logger = logging.getLogger("mp268.api")
logger.setLevel(logging.INFO)
logger.propagate = False
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(JsonFormatter())
logger.handlers = [stdout_handler]

app = FastAPI(title="MP268 Surat Fuel Sales Prediction Service")


def get_model():
    model_path = CFG["paths"]["model_path"]
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = get_model()


class PredictRequest(BaseModel):
    highway_distance_km: float
    nearby_pumps: int
    price_rs: float
    trucks_passing: int = Field(..., le=4000)
    temp_c: float
    staff: float = Field(..., ge=1.5)
    ownership: Literal["public_sector", "private", "cooperative"]
    zone: Literal["Highway", "City", "Industrial"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    active_model = model if model is not None else get_model()
    features_df = pd.DataFrame([request.model_dump()])
    pred_value = float(active_model.predict(features_df)[0])

    logger.info(
        "prediction_completed",
        extra={
            "event_data": {
                "zone": request.zone,
                "ownership": request.ownership,
                "prediction": round(pred_value, 2),
            }
        },
    )
    return {"prediction": pred_value}