import logging
import json

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from schemas import SeverityResponse
from services.gemini_service import analyze_image
from services.image_service import validate_and_read_image

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/severity", response_model=SeverityResponse)
@limiter.limit(settings.rate_limit_gemini)
async def analyze_severity(request: Request, file: UploadFile = File(...)):
    _, image_bytes = await validate_and_read_image(file, settings)

    prompt = """Analyze this potato leaf image for disease severity.

Return ONLY a JSON object with exactly these keys:
- "severity_level": one of "mild", "moderate", or "severe"
- "affected_area_percent": estimated percentage of leaf area affected (number between 0 and 100)
- "urgency": one of "low", "medium", "high", "critical"
- "immediate_actions": array of 2-4 recommended immediate actions (strings)
- "detailed_analysis": a 2-3 sentence analysis of what you observe

Return ONLY the JSON object, no other text."""

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
        logger.exception("Gemini severity analysis failed")
        raise HTTPException(status_code=502, detail="AI severity analysis failed")

    try:
        text = response.strip()
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        data = json.loads(text)
        return SeverityResponse(**data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.warning("Failed to parse Gemini severity response, using fallback")
        return SeverityResponse(
            severity_level="unknown",
            affected_area_percent=0,
            urgency="medium",
            immediate_actions=["Consult a local agricultural expert for accurate assessment"],
            detailed_analysis=f"Automated severity analysis could not be completed. Raw analysis: {response[:300]}",
        )
