import json
from unittest.mock import patch

from fastapi import HTTPException

from tests.conftest import upload_file, _create_test_image_bytes


VALID_SEVERITY_JSON = json.dumps({
    "severity_level": "moderate",
    "affected_area_percent": 35.0,
    "urgency": "high",
    "immediate_actions": ["Remove affected leaves", "Apply fungicide"],
    "detailed_analysis": "Significant infection observed on lower leaves.",
})


class TestSeveritySuccess:
    def test_severity_success(self, client, jpeg_image_bytes):
        with patch("routes.severity.analyze_image", return_value=VALID_SEVERITY_JSON):
            resp = client.post("/severity", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        data = resp.json()
        assert data["severity_level"] == "moderate"
        assert data["affected_area_percent"] == 35.0
        assert data["urgency"] == "high"
        assert len(data["immediate_actions"]) == 2
        assert "infection" in data["detailed_analysis"].lower()

    def test_severity_json_in_markdown(self, client, jpeg_image_bytes):
        md = f"```json\n{VALID_SEVERITY_JSON}\n```"
        with patch("routes.severity.analyze_image", return_value=md):
            resp = client.post("/severity", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        assert resp.json()["severity_level"] == "moderate"


class TestSeverityFallback:
    def test_severity_invalid_json_fallback(self, client, jpeg_image_bytes):
        with patch("routes.severity.analyze_image", return_value="Cannot analyze"):
            resp = client.post("/severity", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        assert resp.json()["severity_level"] == "unknown"
        assert resp.json()["affected_area_percent"] == 0

    def test_severity_fallback_includes_raw(self, client, jpeg_image_bytes):
        with patch("routes.severity.analyze_image", return_value="Some text about the leaf"):
            resp = client.post("/severity", files=[upload_file(jpeg_image_bytes)])
        assert "Some text" in resp.json()["detailed_analysis"]


class TestSeverityErrors:
    def test_severity_unsupported_file_type(self, client):
        data = _create_test_image_bytes()
        resp = client.post("/severity", files=[upload_file(data, content_type="image/gif")])
        assert resp.status_code == 415

    def test_severity_file_too_large(self, client, oversized_image_bytes):
        resp = client.post("/severity", files=[upload_file(oversized_image_bytes)])
        assert resp.status_code == 413

    def test_severity_corrupt_image(self, client):
        resp = client.post("/severity", files=[upload_file(b"garbage")])
        assert resp.status_code == 400

    def test_severity_gemini_503(self, client, jpeg_image_bytes):
        with patch("routes.severity.analyze_image", side_effect=HTTPException(503, "unavailable")):
            resp = client.post("/severity", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 503

    def test_severity_gemini_generic_exception(self, client, jpeg_image_bytes):
        with patch("routes.severity.analyze_image", side_effect=RuntimeError("boom")):
            resp = client.post("/severity", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 502

    def test_severity_no_file(self, client):
        resp = client.post("/severity")
        assert resp.status_code == 422
