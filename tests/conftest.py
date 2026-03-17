import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _create_test_image_bytes(width=256, height=256, color=(0, 128, 0), fmt="JPEG"):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def jpeg_image_bytes():
    return _create_test_image_bytes(fmt="JPEG")


@pytest.fixture(scope="session")
def oversized_image_bytes():
    return b"\x00" * (11 * 1024 * 1024)


@pytest.fixture(scope="session")
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([[0.95, 0.03, 0.02]])
    return model


@pytest.fixture(scope="session")
def mock_gemini_client():
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "mock response"
    client.models.generate_content.return_value = mock_response
    return client


def upload_file(data: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
    return ("file", (filename, io.BytesIO(data), content_type))


def _disable_all_rate_limiters():
    """Disable rate limiters on the main app and all route-level limiters."""
    from main import app, limiter as main_limiter
    from routes import predict, advice, calendar, severity, analyze
    main_limiter.enabled = False
    app.state.limiter.enabled = False
    predict.limiter.enabled = False
    advice.limiter.enabled = False
    calendar.limiter.enabled = False
    severity.limiter.enabled = False
    analyze.limiter.enabled = False


def _enable_all_rate_limiters():
    from main import app, limiter as main_limiter
    from routes import predict, advice, calendar, severity, analyze
    main_limiter.enabled = True
    app.state.limiter.enabled = True
    predict.limiter.enabled = True
    advice.limiter.enabled = True
    calendar.limiter.enabled = True
    severity.limiter.enabled = True
    analyze.limiter.enabled = True


@pytest.fixture()
def client(mock_model, mock_gemini_client):
    with patch("main.load_model"), \
         patch("main.init_gemini"), \
         patch("model_loader._model", mock_model), \
         patch("routes.predict.get_model", return_value=mock_model), \
         patch("services.gemini_service._client", mock_gemini_client):
        from main import app
        _disable_all_rate_limiters()
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c
        _enable_all_rate_limiters()


@pytest.fixture()
def client_no_model(mock_gemini_client):
    with patch("main.load_model"), \
         patch("main.init_gemini"), \
         patch("model_loader._model", None), \
         patch("services.gemini_service._client", mock_gemini_client):
        from main import app
        _disable_all_rate_limiters()
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c
        _enable_all_rate_limiters()


@pytest.fixture()
def client_no_gemini(mock_model):
    with patch("main.load_model"), \
         patch("main.init_gemini"), \
         patch("model_loader._model", mock_model), \
         patch("routes.predict.get_model", return_value=mock_model), \
         patch("services.gemini_service._client", None):
        from main import app
        _disable_all_rate_limiters()
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c
        _enable_all_rate_limiters()
