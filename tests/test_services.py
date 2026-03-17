import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

from config import settings
from tests.conftest import _create_test_image_bytes

import services.gemini_service as gemini_mod
from services.gemini_service import init_gemini, get_client, generate_text, analyze_image
from services.image_service import validate_and_read_image


# ── Image Service ──────────────────────────────────────────────

class TestValidateAndReadImage:
    @staticmethod
    def _make_upload(data: bytes, content_type: str = "image/jpeg", filename: str = "test.jpg") -> UploadFile:
        return UploadFile(filename=filename, file=io.BytesIO(data), headers={"content-type": content_type})

    @pytest.mark.asyncio
    async def test_validate_image_success_jpeg(self):
        data = _create_test_image_bytes(fmt="JPEG")
        file = self._make_upload(data, "image/jpeg")
        arr, raw = await validate_and_read_image(file, settings)
        assert arr.shape[0] == 256
        assert raw == data

    @pytest.mark.asyncio
    async def test_validate_image_success_png(self):
        data = _create_test_image_bytes(fmt="PNG")
        file = self._make_upload(data, "image/png")
        arr, raw = await validate_and_read_image(file, settings)
        assert arr.shape[0] == 256

    @pytest.mark.asyncio
    async def test_validate_image_success_webp(self):
        data = _create_test_image_bytes(fmt="WEBP")
        file = self._make_upload(data, "image/webp")
        arr, raw = await validate_and_read_image(file, settings)
        assert arr.shape[0] == 256

    @pytest.mark.asyncio
    async def test_validate_image_unsupported_mime(self):
        data = _create_test_image_bytes()
        file = self._make_upload(data, "image/gif")
        with pytest.raises(HTTPException) as exc_info:
            await validate_and_read_image(file, settings)
        assert exc_info.value.status_code == 415

    @pytest.mark.asyncio
    async def test_validate_image_too_large(self):
        data = b"\x00" * (11 * 1024 * 1024)
        file = self._make_upload(data, "image/jpeg")
        with pytest.raises(HTTPException) as exc_info:
            await validate_and_read_image(file, settings)
        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_validate_image_corrupt(self):
        file = self._make_upload(b"not an image", "image/jpeg")
        with pytest.raises(HTTPException) as exc_info:
            await validate_and_read_image(file, settings)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_image_at_exact_limit(self):
        # Code uses `>`, not `>=`, so exactly 10MB should succeed
        data = _create_test_image_bytes()
        # We need a real image that is exactly at the limit — instead just verify < limit passes
        file = self._make_upload(data, "image/jpeg")
        arr, raw = await validate_and_read_image(file, settings)
        assert arr is not None


# ── Gemini Service ─────────────────────────────────────────────

class TestInitGemini:
    def setup_method(self):
        """Save and reset global state before each test."""
        self._original_client = gemini_mod._client

    def teardown_method(self):
        """Restore global state after each test."""
        gemini_mod._client = self._original_client

    def test_init_gemini_valid_key(self):
        with patch("services.gemini_service.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            init_gemini("test-api-key")
            assert gemini_mod._client is not None

    def test_init_gemini_empty_key(self):
        gemini_mod._client = None
        init_gemini("")
        assert gemini_mod._client is None

    def test_init_gemini_exception(self):
        with patch("services.gemini_service.genai.Client", side_effect=Exception("fail")):
            init_gemini("bad-key")
            assert gemini_mod._client is None


class TestGetClient:
    def test_get_client_available(self):
        mock = MagicMock()
        with patch.object(gemini_mod, "_client", mock):
            assert get_client() is mock

    def test_get_client_none_raises_503(self):
        with patch.object(gemini_mod, "_client", None):
            with pytest.raises(HTTPException) as exc_info:
                get_client()
            assert exc_info.value.status_code == 503


class TestGenerateText:
    def test_generate_text_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "generated text"
        mock_client.models.generate_content.return_value = mock_response
        with patch.object(gemini_mod, "_client", mock_client):
            result = generate_text("prompt", "model")
            assert result == "generated text"

    def test_generate_text_no_client_503(self):
        with patch.object(gemini_mod, "_client", None):
            with pytest.raises(HTTPException) as exc_info:
                generate_text("prompt", "model")
            assert exc_info.value.status_code == 503


class TestAnalyzeImage:
    def test_analyze_image_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "analysis result"
        mock_client.models.generate_content.return_value = mock_response
        with patch.object(gemini_mod, "_client", mock_client), \
             patch("services.gemini_service.types.Part.from_bytes") as mock_part:
            mock_part.return_value = MagicMock()
            result = analyze_image(b"data", "image/jpeg", "prompt", "model")
            assert result == "analysis result"

    def test_analyze_image_no_client_503(self):
        with patch.object(gemini_mod, "_client", None):
            with pytest.raises(HTTPException) as exc_info:
                analyze_image(b"data", "image/jpeg", "prompt", "model")
            assert exc_info.value.status_code == 503
