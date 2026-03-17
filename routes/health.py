from fastapi import APIRouter

import model_loader
import services.gemini_service as gemini_mod
from schemas import HealthResponse

router = APIRouter()


@router.get("/ping")
async def ping():
    return "Hello, I am alive"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model_loader._model is not None,
        gemini_available=gemini_mod._client is not None,
    )
