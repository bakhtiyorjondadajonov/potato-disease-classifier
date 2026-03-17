import logging

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image
from io import BytesIO

from config import Settings

logger = logging.getLogger(__name__)


async def validate_and_read_image(
    file: UploadFile, settings: Settings
) -> tuple[np.ndarray, bytes]:
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {settings.allowed_content_types}",
        )

    data = await file.read()

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    try:
        image = np.array(Image.open(BytesIO(data)))
    except Exception:
        logger.exception("Failed to read image")
        raise HTTPException(status_code=400, detail="Could not read image file. It may be corrupt.")

    return image, data
