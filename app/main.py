# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from .model_loader import load_model

app = FastAPI(title="Enterprise MLOps API", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: int
    probabilities: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # validate feature length
    if len(req.features) != 4:
        # contoh bug di skenario hotfix: sebelumnya return 500
        raise HTTPException(
            status_code=400,
            detail="features must have length 4"
        )

    model = load_model()
    X = np.array(req.features).reshape(1, -1)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0].tolist()

    return PredictResponse(
        prediction=int(pred),
        probabilities=probs
    )
