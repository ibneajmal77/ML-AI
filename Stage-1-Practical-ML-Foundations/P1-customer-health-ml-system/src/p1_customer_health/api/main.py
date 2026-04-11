from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from p1_customer_health.api.schemas import PredictRequest, PredictResponse
from p1_customer_health.app.settings import get_settings
from p1_customer_health.serving.prediction_service import PredictionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading prediction service...")
    app.state.prediction_service = PredictionService.from_settings(get_settings())
    logger.info("Prediction service ready. model_version=%s", app.state.prediction_service.model_version)
    yield
    logger.info("Shutting down prediction service.")


app = FastAPI(title="P1 Customer Health ML System", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    service: PredictionService | None = getattr(app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok", "model_version": service.model_version}


@app.post("/v1/predict/customer-health", response_model=PredictResponse)
def predict_customer_health(request: PredictRequest) -> PredictResponse:
    service: PredictionService = app.state.prediction_service
    try:
        predictions = service.predict([record.model_dump() for record in request.records])
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.") from exc
    return PredictResponse(model_version=service.model_version, predictions=predictions)
