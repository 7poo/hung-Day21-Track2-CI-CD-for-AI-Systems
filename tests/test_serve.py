import importlib
import joblib

from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier


def _reload_serve_module(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pkl"
    X = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    ]
    y = [0, 1, 2, 2]

    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    joblib.dump(model, model_path)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.delenv("GCS_BUCKET", raising=False)

    import src.serve as serve
    return importlib.reload(serve)


def test_health_endpoint(tmp_path, monkeypatch):
    serve = _reload_serve_module(tmp_path, monkeypatch)
    client = TestClient(serve.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(tmp_path, monkeypatch):
    serve = _reload_serve_module(tmp_path, monkeypatch)
    client = TestClient(serve.app)

    response = client.post("/predict", json={"features": [0.1] * 12})

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data
