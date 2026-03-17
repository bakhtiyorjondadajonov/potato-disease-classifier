import logging

import keras
import tensorflow as tf
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Patch RandomRotation to accept the removed `value_range` kwarg
# so models saved with older Keras versions can still be loaded.
_orig_rr_init = keras.layers.RandomRotation.__init__


def _compat_rr_init(self, *args, **kwargs):
    kwargs.pop("value_range", None)
    _orig_rr_init(self, *args, **kwargs)


keras.layers.RandomRotation.__init__ = _compat_rr_init

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
