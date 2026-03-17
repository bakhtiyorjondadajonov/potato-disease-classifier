import logging

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from model_loader import get_model, CLASS_NAMES
from schemas import PredictionResponse
from services.image_service import validate_and_read_image

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit(settings.rate_limit_predict)
async def predict(request: Request, file: UploadFile = File(...)):
    image, _ = await validate_and_read_image(file, settings)

    model = get_model()
    image_batch = np.expand_dims(image, 0)

    try:
        predictions = model.predict(image_batch)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    result_ind = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[result_ind]
    confidence = float(np.max(predictions[0]))

    logger.info("Prediction: %s (%.2f%%)", predicted_class, confidence * 100)

    return PredictionResponse(
        **{
            "class": predicted_class,
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.1f}%",
        }
    )
