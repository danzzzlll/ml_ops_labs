from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info_has_keys():
    r = client.get("/model_info")
    assert r.status_code == 200
    data = r.json()
    for k in ("model_name", "framework", "labels", "pipeline_task"):
        assert k in data
    assert isinstance(data["labels"], list) and len(data["labels"]) >= 1


def test_predict_returns_label_and_score():
    r = client.post("/predict", json={"text": "Это абсолютно нейтральный текст."})
    assert r.status_code == 200
    data = r.json()
    assert "label" in data and "score" in data
    assert isinstance(data["score"], float)
    assert 0.0 <= data["score"] <= 1.0


def test_predict_batch_same_length():
    texts = ["Текст номер один.", "Ты очень плохой человек!", "Просто погода хорошая"]
    r = client.post("/predict_batch", json={"texts": texts})
    assert r.status_code == 200
    data = r.json()
    preds = data["predictions"]
    assert isinstance(preds, list)
    assert len(preds) == len(texts)
    for p in preds:
        assert "label" in p and "score" in p


def test_validation_error_on_empty_text():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422
