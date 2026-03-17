import logging
import json

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from schemas import PlantType, AnalyzeResponse
from prompts.plant_prompts import get_plant_prompt
from services.gemini_service import analyze_image
from services.image_service import validate_and_read_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Plant Analysis"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/{plant_type}", response_model=AnalyzeResponse)
@limiter.limit(settings.rate_limit_gemini)
async def analyze_plant(
    request: Request,
    plant_type: PlantType,
    file: UploadFile = File(...),
):
    """Analyze a plant leaf image for disease using Gemini vision with expert prompts."""
    _, image_bytes = await validate_and_read_image(file, settings)

    prompt = get_plant_prompt(plant_type.value)

    try:
        response = analyze_image(
            image_bytes,
            file.content_type or "image/jpeg",
            prompt,
            settings.gemini_model,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Gemini plant analysis failed for %s", plant_type.value)
        raise HTTPException(status_code=502, detail="AI plant analysis failed")

    try:
        text = response.strip()
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        data = json.loads(text)
        return AnalyzeResponse(
            plant_type=plant_type.value,
            disease_name=data["disease_name"],
            is_healthy=data["is_healthy"],
            confidence=data["confidence"],
            description=data["description"],
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.warning("Failed to parse Gemini analysis response for %s, using fallback", plant_type.value)
        return AnalyzeResponse(
            plant_type=plant_type.value,
            disease_name="Not Identified",
            is_healthy=False,
            confidence=0.0,
            description=f"Automated analysis could not be completed. Raw response: {response[:300]}",
        )
