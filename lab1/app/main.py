from app.model import model_info, predict_batch, predict_one
from app.schemas import (
    BatchPredictionOut,
    BatchTextIn,
    HealthOut,
    ModelInfoOut,
    PredictionOut,
    TextIn,
)
from fastapi import FastAPI

app = FastAPI(title="Toxicity API (RU)", version="1.0.0")


@app.get("/health", response_model=HealthOut)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    return predict_one(payload.text)


@app.post("/predict_batch", response_model=BatchPredictionOut)
def predict_batch_endpoint(payload: BatchTextIn):
    preds = predict_batch(payload.texts)
    return {"predictions": preds}


@app.get("/model_info", response_model=ModelInfoOut)
def model_info_endpoint():
    return model_info()
