import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(small_df, test_config, monkeypatch):
    from energy_forecast.train import train
    train(test_config, df=small_df)

    import energy_forecast.api as api
    from pathlib import Path
    monkeypatch.setattr(api, "MODEL_PATH", Path(test_config["artifacts"]["model_path"]))
    monkeypatch.setattr(api, "METRICS_PATH", Path(test_config["artifacts"]["metrics_path"]))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict(client):
    payload = [{
        "timestamp": "2024-02-20T10:00:00", "building_type": "office",
        "temperature": 5.0, "lag_24": 90.0, "lag_168": 92.0, "rolling_mean_24": 55.0,
    }]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()[0]["predicted_consumption_kwh"] > 0


def test_rejects_unknown_building_type(client):
    payload = [{
        "timestamp": "2024-02-20T10:00:00", "building_type": "factory",
        "temperature": 5.0, "lag_24": 90.0, "lag_168": 92.0, "rolling_mean_24": 55.0,
    }]
    assert client.post("/predict", json=payload).status_code == 422
