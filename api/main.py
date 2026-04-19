"""
api/main.py

FastAPI service wrapping the trained churn model.
Run with: uvicorn api.main:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, HTTPException, status

from api.predictor import ChurnPredictor, get_predictor
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    OrderFeatures,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model once at startup so the first request doesn't pay the cost
    try:
        get_predictor()
        logger.info("Model loaded and ready.")
    except FileNotFoundError as e:
        logger.error("Couldn't find model artifacts: %s", e)
        logger.error("Run python scripts/train.py first.")
    yield


app = FastAPI(
    title="Churn Predictor",
    description=(
        "Predicts repeat purchase probability from first-order features. "
        "Trained on the Olist Brazilian e-commerce dataset using FLAML AutoML."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo, tags=["ops"])
def model_info(predictor: ChurnPredictor = Depends(get_predictor)):
    return ModelInfo(
        variant=predictor.estimator_name,
        feature_count=len(predictor.feature_names),
        feature_names=predictor.feature_names,
        estimator=predictor.estimator_name,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(order: OrderFeatures, predictor: ChurnPredictor = Depends(get_predictor)):
    """
    Score a single delivered order.
    Returns churn probability + binary label (1 = likely one-time buyer).
    """
    try:
        return PredictionResponse(**predictor.predict(order.model_dump()))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(request: BatchPredictionRequest, predictor: ChurnPredictor = Depends(get_predictor)):
    """
    Score a list of orders in one shot — useful for bulk CRM exports.
    """
    if not request.orders:
        raise HTTPException(status_code=422, detail="orders list is empty")
    try:
        results = predictor.predict_batch([o.model_dump() for o in request.orders])
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
