import logging
import time

from google import genai
from google.genai import types
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

_client = None

RETRY_DELAY = 2
MAX_RETRIES = 1


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


def _retry_on_rate_limit(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            if e.status_code == 429 and attempt < MAX_RETRIES:
                logger.warning("Rate limited by Gemini API, retrying in %ds...", RETRY_DELAY)
                time.sleep(RETRY_DELAY)
            else:
                raise


def generate_text(prompt: str, model: str) -> str:
    client = get_client()

    def _call():
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text

    return _retry_on_rate_limit(_call)


def analyze_image(image_bytes: bytes, mime_type: str, prompt: str, model: str) -> str:
    client = get_client()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image_part],
        )
        return response.text

    return _retry_on_rate_limit(_call)
