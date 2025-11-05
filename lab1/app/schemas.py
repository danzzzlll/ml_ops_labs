from typing import List

from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    status: str = "ok"


class TextIn(BaseModel):
    text: str = Field(..., min_length=1)


class BatchTextIn(BaseModel):
    texts: List[str] = Field(..., min_items=1)


class PredictionOut(BaseModel):
    label: str
    score: float


class BatchPredictionOut(BaseModel):
    predictions: List[PredictionOut]


class ModelInfoOut(BaseModel):
    model_name: str
    framework: str
    labels: List[str]
    pipeline_task: str
