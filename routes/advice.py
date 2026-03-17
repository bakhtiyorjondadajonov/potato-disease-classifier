import logging

from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from schemas import AdviceRequest, AdviceResponse
from services.gemini_service import generate_text

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/advice", response_model=AdviceResponse)
@limiter.limit(settings.rate_limit_gemini)
async def get_advice(request: FastAPIRequest, body: AdviceRequest):
    plant = body.plant_type.value if body.plant_type else "potato"
    prompt = f"""You are an expert agricultural advisor specializing in {plant} diseases.

The disease detected is: {body.disease} (confidence: {body.confidence * 100:.1f}%)

Provide a response in EXACTLY this format (use these exact headers):

TREATMENT:
[Detailed treatment recommendations for this specific {plant} disease]

PREVENTION:
[Prevention tips to avoid future occurrences]

CARE INSTRUCTIONS:
[Ongoing care instructions for the affected {plant} plants]

Be specific, practical, and actionable. Keep each section to 2-4 sentences."""

    try:
        response = generate_text(prompt, settings.gemini_model)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Gemini advice generation failed")
        raise HTTPException(status_code=502, detail="AI advice generation failed")

    sections = {"treatment": "", "prevention": "", "care_instructions": ""}
    current_section = None

    for line in response.split("\n"):
        line_stripped = line.strip()
        upper = line_stripped.upper()
        if upper.startswith("TREATMENT"):
            current_section = "treatment"
        elif upper.startswith("PREVENTION"):
            current_section = "prevention"
        elif upper.startswith("CARE INSTRUCTIONS") or upper.startswith("CARE_INSTRUCTIONS"):
            current_section = "care_instructions"
        elif current_section and line_stripped:
            sections[current_section] += line_stripped + " "

    for key in sections:
        sections[key] = sections[key].strip() or "No information available."

    return AdviceResponse(disease=body.disease, **sections)
