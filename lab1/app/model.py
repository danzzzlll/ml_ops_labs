import os
from functools import lru_cache
from typing import List

from transformers import pipeline

MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "models/",
)


@lru_cache(maxsize=1)
def get_pipeline():
    clf = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        top_k=None,
    )
    return clf


def predict_one(text: str):
    clf = get_pipeline()
    out = clf(text)
    pred = out[0]
    return pred[0]


def predict_batch(texts: List[str]):
    clf = get_pipeline()
    outs = clf(texts)
    results = []
    for o in outs:
        pred = o[0] if isinstance(o, list) else o
        results.append({"label": str(pred["label"]), "score": float(pred["score"])})
    return results


def model_info():
    clf = get_pipeline()
    config = clf.model.config
    labels = list(getattr(config, "id2label", {}).values()) or ["non-toxic", "toxic"]
    return {
        "model_name": getattr(
            config, "name_or_path", "cointegrated/rubert-tiny-toxicity"
        ),
        "framework": "transformers/torch",
        "labels": labels,
        "pipeline_task": "text-classification",
    }
