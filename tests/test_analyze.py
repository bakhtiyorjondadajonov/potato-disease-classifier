import json
from unittest.mock import patch

from fastapi import HTTPException

from tests.conftest import upload_file, _create_test_image_bytes


def _valid_analyze_json(disease="Early Blight", is_healthy=False, confidence=0.85):
    return json.dumps({
        "disease_name": disease,
        "is_healthy": is_healthy,
        "confidence": confidence,
        "description": "Visible symptoms detected on the leaf.",
    })


class TestAnalyzeSuccess:
    def test_analyze_tomato_success(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value=_valid_analyze_json()):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        assert resp.json()["plant_type"] == "tomato"

    def test_analyze_corn_success(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value=_valid_analyze_json()):
            resp = client.post("/analyze/corn", files=[upload_file(jpeg_image_bytes)])
        assert resp.json()["plant_type"] == "corn"

    def test_analyze_pepper_success(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value=_valid_analyze_json()):
            resp = client.post("/analyze/pepper", files=[upload_file(jpeg_image_bytes)])
        assert resp.json()["plant_type"] == "pepper"

    def test_analyze_apple_success(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value=_valid_analyze_json()):
            resp = client.post("/analyze/apple", files=[upload_file(jpeg_image_bytes)])
        assert resp.json()["plant_type"] == "apple"

    def test_analyze_strawberry_success(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value=_valid_analyze_json()):
            resp = client.post("/analyze/strawberry", files=[upload_file(jpeg_image_bytes)])
        assert resp.json()["plant_type"] == "strawberry"

    def test_analyze_healthy_plant(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value=_valid_analyze_json("Healthy", True, 0.95)):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.json()["is_healthy"] is True


class TestAnalyzeParsing:
    def test_analyze_json_in_markdown(self, client, jpeg_image_bytes):
        md = f"```json\n{_valid_analyze_json()}\n```"
        with patch("routes.analyze.analyze_image", return_value=md):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        assert resp.json()["disease_name"] == "Early Blight"

    def test_analyze_invalid_json_fallback(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value="garbage"):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        assert resp.json()["disease_name"] == "Not Identified"
        assert resp.json()["confidence"] == 0.0

    def test_analyze_missing_key_fallback(self, client, jpeg_image_bytes):
        partial = json.dumps({"disease_name": "X"})
        with patch("routes.analyze.analyze_image", return_value=partial):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.json()["disease_name"] == "Not Identified"

    def test_analyze_fallback_includes_raw(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value="Raw text from gemini"):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert "Raw text" in resp.json()["description"]

    def test_analyze_confidence_boundary(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value=_valid_analyze_json(confidence=1.0)):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200

    def test_analyze_invalid_confidence_from_gemini(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", return_value=_valid_analyze_json(confidence=1.5)):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        # Pydantic rejects confidence > 1.0 → fallback
        assert resp.json()["disease_name"] == "Not Identified"


class TestAnalyzeErrors:
    def test_analyze_invalid_plant_type(self, client, jpeg_image_bytes):
        resp = client.post("/analyze/banana", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 422

    def test_analyze_unsupported_file_type(self, client):
        data = _create_test_image_bytes()
        resp = client.post("/analyze/tomato", files=[upload_file(data, content_type="image/gif")])
        assert resp.status_code == 415

    def test_analyze_file_too_large(self, client, oversized_image_bytes):
        resp = client.post("/analyze/tomato", files=[upload_file(oversized_image_bytes)])
        assert resp.status_code == 413

    def test_analyze_gemini_503(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", side_effect=HTTPException(503, "unavailable")):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 503

    def test_analyze_gemini_generic_exception(self, client, jpeg_image_bytes):
        with patch("routes.analyze.analyze_image", side_effect=RuntimeError("boom")):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 502

    def test_analyze_no_file(self, client):
        resp = client.post("/analyze/tomato")
        assert resp.status_code == 422
