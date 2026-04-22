from fastapi import FastAPI

from app.routers.classify import router as classify_router
from app.routers.extract import router as extract_router
from app.routers.route import router as route_router


app = FastAPI(title="ticket-ops-api", version="0.1.0")
app.include_router(classify_router)
app.include_router(extract_router)
app.include_router(route_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
