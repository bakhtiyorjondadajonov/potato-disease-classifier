import logging
import json

from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from schemas import CropCalendarRequest, CropCalendarResponse, ActionItem
from services.gemini_service import generate_text

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/crop-calendar", response_model=CropCalendarResponse)
@limiter.limit(settings.rate_limit_gemini)
async def get_crop_calendar(request: FastAPIRequest, body: CropCalendarRequest):
    season = body.season or "current"
    location = body.location or "general"

    plant = body.plant_type or "potato"
    prompt = f"""You are an expert agricultural planner for {plant} crops.

Disease detected: {body.disease} (confidence: {body.confidence * 100:.1f}%)
Season: {season}
Location: {location}

Generate a personalized action timeline as a JSON array. Each item must have exactly these keys:
- "week": a time label like "Week 1", "Week 2-3", etc.
- "action": a short action title
- "details": specific instructions

Return ONLY a JSON array with 4-6 items, no other text. Example format:
[{{"week": "Week 1", "action": "Initial Treatment", "details": "Apply fungicide..."}}]"""

    try:
        response = generate_text(prompt, settings.gemini_model)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Gemini crop calendar generation failed")
        raise HTTPException(status_code=502, detail="AI calendar generation failed")

    try:
        text = response.strip()
        # Extract JSON array from response (handle markdown code blocks)
        if "```" in text:
            start = text.find("[")
            end = text.rfind("]") + 1
            text = text[start:end]
        timeline_data = json.loads(text)
        timeline = [ActionItem(**item) for item in timeline_data]
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Failed to parse Gemini calendar response, using fallback")
        timeline = [
            ActionItem(
                week="Week 1",
                action="Assessment",
                details=f"Assess the extent of {body.disease} in your potato crop.",
            ),
            ActionItem(
                week="Week 2",
                action="Treatment",
                details=f"Apply recommended treatment for {body.disease}.",
            ),
            ActionItem(
                week="Week 3-4",
                action="Monitoring",
                details="Monitor plants for improvement and reapply treatment if needed.",
            ),
        ]

    return CropCalendarResponse(
        disease=body.disease,
        season=season,
        timeline=timeline,
    )
