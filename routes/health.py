from fastapi import APIRouter

from model_loader import _model
from schemas import HealthResponse

router = APIRouter()


@router.get("/ping")
async def ping():
    return "Hello, I am alive"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", model_loaded=_model is not None)
