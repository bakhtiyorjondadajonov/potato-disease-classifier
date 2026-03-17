import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_client = None


def init_gemini(api_key: str):
    global _client
    if not api_key:
        logger.warning("No Gemini API key provided. Gemini features will be unavailable.")
        return
    try:
        _client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized")
    except Exception as e:
        logger.error("Failed to initialize Gemini client: %s", e)
        _client = None


def get_client() -> genai.Client:
    if _client is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Gemini service not available. Check API key configuration.")
    return _client


def generate_text(prompt: str, model: str) -> str:
    client = get_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def analyze_image(image_bytes: bytes, mime_type: str, prompt: str, model: str) -> str:
    client = get_client()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image_part],
    )
    return response.text
