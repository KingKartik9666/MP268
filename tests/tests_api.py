import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

import app.main as app_main
from src.mp268.checks import check_unit_mixup, check_stuck_sensor, load_params
from src.mp268.clean import clean_dataframe


@pytest.fixture(autouse=True)
def mock_champion_model(monkeypatch):
    dummy = DummyRegressor(strategy="constant", constant=18500.0)
    dummy.fit([[0]], [18500.0])
    monkeypatch.setattr(app_main, "model", dummy)


client = TestClient(app_main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_and_json_logging(capsys):
    payload = {
        "highway_distance_km": 5.2,
        "nearby_pumps": 4,
        "price_rs": 98.5,
        "trucks_passing": 1200,
        "temp_c": 31.0,
        "staff": 8.0,
        "ownership": "private",
        "zone": "Highway",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(18500.0)

    captured = capsys.readouterr()
    log_line = captured.out.strip().splitlines()[-1]
    parsed = json.loads(log_line)
    assert parsed["message"] == "prediction_completed"
    assert parsed["zone"] == "Highway"


def test_predict_rejects_trucks_above_4000():
    payload = {
        "highway_distance_km": 5.2,
        "nearby_pumps": 4,
        "price_rs": 98.5,
        "trucks_passing": 4001,
        "temp_c": 31.0,
        "staff": 8.0,
        "ownership": "private",
        "zone": "Highway",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_staff_below_1_5():
    payload = {
        "highway_distance_km": 5.2,
        "nearby_pumps": 4,
        "price_rs": 98.5,
        "trucks_passing": 1200,
        "temp_c": 31.0,
        "staff": 1.4,
        "ownership": "private",
        "zone": "Highway",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_checks_and_clean_pipeline():
    cfg = load_params()
    synthetic_df = pd.DataFrame([
        {
            "record_id": "R1",
            "date": "2026-01-01",
            "highway_distance_km": 5.0,
            "nearby_pumps": 3,
            "price_rs": 95.0,
            "trucks_passing": 900,
            "temp_c": 25.0,
            "staff": 9999,
            "ownership": "PUB.",
            "zone": "Highway",
            "litres_from_tank_dip_next_day": 15010,
            "litres_per_day": 15000,
        },
        {
            "record_id": "R2",
            "date": "2026-01-02",
            "highway_distance_km": 6000.0,
            "nearby_pumps": 4,
            "price_rs": 96.0,
            "trucks_passing": 1100,
            "temp_c": 26.0,
            "staff": 10,
            "ownership": " cooperative ",
            "zone": "Industrial",
            "litres_from_tank_dip_next_day": 21020,
            "litres_per_day": 21000,
        },
    ])

    mixup = check_unit_mixup(synthetic_df, ["highway_distance_km"])
    assert mixup["present"] is True
    assert mixup["details"]["factor"] == 1000

    stuck = check_stuck_sensor(synthetic_df, ["highway_distance_km"])
    assert stuck["present"] is False

    cleaned = clean_dataframe(synthetic_df, cfg, is_train=True)
    assert "litres_from_tank_dip_next_day" not in cleaned.columns
    assert cleaned.loc[1, "highway_distance_km"] == pytest.approx(6.0)
    assert pd.isna(cleaned.loc[0, "staff"])
    assert list(cleaned["ownership"]) == ["public_sector", "cooperative"]