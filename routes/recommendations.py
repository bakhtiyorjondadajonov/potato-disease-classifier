import logging
import json

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, Form
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from schemas import (
    RecommendationsResponse,
    SeverityResponse,
    AdviceResponse,
    ActionItem,
    CropCalendarResponse,
)
from services.gemini_service import analyze_image
from services.image_service import validate_and_read_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Recommendations"])
limiter = Limiter(key_func=get_remote_address)


def _build_prompt(disease: str, confidence: float, plant: str, season: str, location: str) -> str:
    return f"""You are an expert agricultural advisor specializing in {plant} diseases.

The disease detected is: {disease} (confidence: {confidence * 100:.1f}%)
Season: {season}
Location: {location}

Analyze the provided leaf image and return a comprehensive JSON response with ALL of the following sections:

Return ONLY a JSON object with exactly these keys:

1. Severity assessment:
- "severity_level": one of "mild", "moderate", or "severe"
- "affected_area_percent": estimated percentage of leaf area affected (number between 0 and 100)
- "urgency": one of "low", "medium", "high", "critical"
- "immediate_actions": array of 2-4 recommended immediate actions (strings)
- "detailed_analysis": a 2-3 sentence analysis of what you observe

2. Treatment advice:
- "treatment": 2-4 sentences of detailed treatment recommendations
- "prevention": 2-4 sentences of prevention tips
- "care_instructions": 2-4 sentences of ongoing care instructions

3. Action timeline (for the given season and location):
- "season": the current season
- "timeline": array of 4-6 objects, each with "week" (e.g. "Week 1"), "action" (short title), "details" (1-2 sentences)

Return ONLY the JSON object, no other text or markdown."""


@router.post("/recommendations", response_model=RecommendationsResponse)
@limiter.limit(settings.rate_limit_gemini)
async def get_recommendations(
    request: Request,
    file: UploadFile = File(...),
    disease: str = Form(...),
    confidence: float = Form(...),
    plant_type: str = Form("potato"),
    season: str = Form("Summer"),
    location: str = Form("Poland"),
):
    """Get severity, advice, and calendar in a single Gemini call."""
    _, image_bytes = await validate_and_read_image(file, settings)

    prompt = _build_prompt(disease, confidence, plant_type, season, location)

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
        logger.exception("Gemini recommendations failed for %s", plant_type)
        raise HTTPException(status_code=502, detail="AI recommendations generation failed")

    try:
        text = response.strip()
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        data = json.loads(text)

        # Gemini may return nested structure with section keys or a flat dict.
        # Flatten: merge all nested dicts into one flat dict.
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        severity = SeverityResponse(
            severity_level=flat.get("severity_level", "unknown"),
            affected_area_percent=flat.get("affected_area_percent", 0),
            urgency=flat.get("urgency", "medium"),
            immediate_actions=flat.get("immediate_actions", ["Consult a local agricultural expert"]),
            detailed_analysis=flat.get("detailed_analysis", "Analysis not available."),
        )

        advice = AdviceResponse(
            disease=disease,
            treatment=flat.get("treatment", "No information available."),
            prevention=flat.get("prevention", "No information available."),
            care_instructions=flat.get("care_instructions", "No information available."),
        )

        timeline_raw = flat.get("timeline", [])
        timeline = []
        for item in timeline_raw:
            if isinstance(item, dict):
                timeline.append(ActionItem(
                    week=item.get("week", ""),
                    action=item.get("action", ""),
                    details=item.get("details", ""),
                ))

        calendar = CropCalendarResponse(
            disease=disease,
            season=flat.get("season", season),
            timeline=timeline if timeline else [
                ActionItem(week="Week 1", action="Assess", details="Evaluate the extent of the disease and plan treatment.")
            ],
        )

        return RecommendationsResponse(
            severity=severity,
            advice=advice,
            calendar=calendar,
        )

    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.warning("Failed to parse Gemini recommendations response, using fallback")
        return RecommendationsResponse(
            severity=SeverityResponse(
                severity_level="unknown",
                affected_area_percent=0,
                urgency="medium",
                immediate_actions=["Consult a local agricultural expert for accurate assessment"],
                detailed_analysis=f"Automated analysis could not be completed. Raw response: {response[:300]}",
            ),
            advice=AdviceResponse(
                disease=disease,
                treatment="Automated advice generation failed. Please consult a local agricultural expert.",
                prevention="No information available.",
                care_instructions="No information available.",
            ),
            calendar=CropCalendarResponse(
                disease=disease,
                season=season,
                timeline=[ActionItem(week="Week 1", action="Consult Expert", details="Seek local agricultural advice for proper treatment plan.")],
            ),
        )
