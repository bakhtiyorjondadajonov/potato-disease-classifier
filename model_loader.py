import logging

import tensorflow as tf
from fastapi import HTTPException

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

_model = None


def load_model(path: str):
    global _model
    try:
        logger.info("Loading TensorFlow model from %s", path)
        _model = tf.keras.models.load_model(path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        _model = None


def get_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model
