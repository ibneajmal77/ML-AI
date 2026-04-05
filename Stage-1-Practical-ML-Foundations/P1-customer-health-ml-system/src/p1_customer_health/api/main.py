from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from p1_customer_health.api.schemas import PredictRequest, PredictResponse
from p1_customer_health.api.service import PredictionService
from p1_customer_health.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = PredictionService.from_settings(get_settings())
    yield


app = FastAPI(title="P1 Customer Health ML System", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/predict/customer-health", response_model=PredictResponse)
def predict_customer_health(request: PredictRequest) -> PredictResponse:
    service: PredictionService = app.state.prediction_service
    predictions = service.predict([record.model_dump() for record in request.records])
    return PredictResponse(model_version="p1_customer_health_v1", predictions=predictions)
