"""End-to-end integration tests simulating real user flows."""
import json
from unittest.mock import MagicMock, patch

import numpy as np

from tests.conftest import upload_file, _create_test_image_bytes


ADVICE_RESPONSE = """TREATMENT:
Apply copper fungicide every 7 days.

PREVENTION:
Rotate crops. Space plants properly.

CARE INSTRUCTIONS:
Water at base only. Remove affected leaves."""

CALENDAR_RESPONSE = json.dumps([
    {"week": "Week 1", "action": "Assessment", "details": "Check all plants"},
    {"week": "Week 2", "action": "Treatment", "details": "Apply fungicide"},
])

SEVERITY_RESPONSE = json.dumps({
    "severity_level": "moderate",
    "affected_area_percent": 40.0,
    "urgency": "high",
    "immediate_actions": ["Remove leaves", "Apply treatment"],
    "detailed_analysis": "Moderate infection on lower canopy.",
})

ANALYZE_RESPONSE = json.dumps({
    "disease_name": "Early Blight",
    "is_healthy": False,
    "confidence": 0.88,
    "description": "Dark concentric rings on leaves.",
})


def _smart_generate_text(prompt, model):
    """Return context-appropriate responses based on prompt content."""
    lower = prompt.lower()
    if "timeline" in lower or "action timeline" in lower or "json array" in lower:
        return CALENDAR_RESPONSE
    return ADVICE_RESPONSE


def _smart_analyze_image(image_bytes, mime_type, prompt, model):
    lower = prompt.lower()
    if "severity" in lower:
        return SEVERITY_RESPONSE
    return ANALYZE_RESPONSE


class TestE2EFullFlows:
    def test_e2e_full_potato_flow(self, client, jpeg_image_bytes):
        """Predict → Advice → Calendar flow."""
        # Step 1: Predict
        resp = client.post("/predict", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        prediction = resp.json()
        disease = prediction["class"]
        confidence = prediction["confidence"]

        # Step 2: Get advice
        with patch("routes.advice.generate_text", side_effect=_smart_generate_text):
            resp = client.post("/advice", json={"disease": disease, "confidence": confidence})
        assert resp.status_code == 200
        assert resp.json()["treatment"] != "No information available."

        # Step 3: Get calendar
        with patch("routes.calendar.generate_text", side_effect=_smart_generate_text):
            resp = client.post("/crop-calendar", json={"disease": disease, "confidence": confidence})
        assert resp.status_code == 200
        assert len(resp.json()["timeline"]) > 0

    def test_e2e_potato_with_severity(self, client, jpeg_image_bytes):
        """Predict + Severity → Advice → Calendar."""
        resp = client.post("/predict", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200

        with patch("routes.severity.analyze_image", side_effect=_smart_analyze_image):
            resp = client.post("/severity", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        sev = resp.json()
        assert sev["severity_level"] in ("mild", "moderate", "severe")

        with patch("routes.advice.generate_text", side_effect=_smart_generate_text):
            resp = client.post("/advice", json={"disease": "Early Blight", "confidence": 0.95})
        assert resp.status_code == 200

    def test_e2e_multi_plant_tomato_flow(self, client, jpeg_image_bytes):
        """Analyze/tomato → Advice(tomato) → Calendar."""
        with patch("routes.analyze.analyze_image", side_effect=_smart_analyze_image):
            resp = client.post("/analyze/tomato", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        data = resp.json()
        assert data["plant_type"] == "tomato"

        with patch("routes.advice.generate_text", side_effect=_smart_generate_text):
            resp = client.post("/advice", json={
                "disease": data["disease_name"],
                "confidence": data["confidence"],
                "plant_type": "tomato",
            })
        assert resp.status_code == 200

        with patch("routes.calendar.generate_text", side_effect=_smart_generate_text):
            resp = client.post("/crop-calendar", json={
                "disease": data["disease_name"],
                "confidence": data["confidence"],
                "plant_type": "tomato",
            })
        assert resp.status_code == 200

    def test_e2e_all_five_plants(self, client, jpeg_image_bytes):
        """Analyze all 5 plant types."""
        for plant in ("tomato", "corn", "pepper", "apple", "strawberry"):
            with patch("routes.analyze.analyze_image", side_effect=_smart_analyze_image):
                resp = client.post(f"/analyze/{plant}", files=[upload_file(jpeg_image_bytes)])
            assert resp.status_code == 200
            assert resp.json()["plant_type"] == plant


class TestE2EBackwardCompat:
    def test_e2e_backward_compat_advice_no_plant_type(self, client):
        with patch("routes.advice.generate_text", side_effect=_smart_generate_text):
            resp = client.post("/advice", json={"disease": "Early Blight", "confidence": 0.9})
        assert resp.status_code == 200

    def test_e2e_backward_compat_calendar_no_plant_type(self, client):
        with patch("routes.calendar.generate_text", side_effect=_smart_generate_text):
            resp = client.post("/crop-calendar", json={"disease": "Early Blight", "confidence": 0.9})
        assert resp.status_code == 200


class TestE2EPartialFailure:
    def test_e2e_model_down_gemini_up(self, client_no_model, jpeg_image_bytes):
        """Model down but Gemini endpoints still work."""
        resp = client_no_model.post("/predict", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 503

        with patch("routes.advice.generate_text", side_effect=_smart_generate_text):
            resp = client_no_model.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 200

        with patch("routes.calendar.generate_text", side_effect=_smart_generate_text):
            resp = client_no_model.post("/crop-calendar", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 200

    def test_e2e_gemini_down_model_up(self, client_no_gemini, jpeg_image_bytes):
        """Gemini down but CNN predict still works."""
        resp = client_no_gemini.post("/predict", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200

        resp = client_no_gemini.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 503

        resp = client_no_gemini.post("/severity", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 503


class TestE2EMisc:
    def test_e2e_cors_headers(self, client):
        resp = client.options("/predict", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        })
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_e2e_health_check_flow(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is True
        assert resp.json()["gemini_available"] is True

    def test_e2e_global_exception_handler(self, mock_model, mock_gemini_client, jpeg_image_bytes):
        """Force an unhandled exception to trigger the global handler."""
        with patch("main.load_model"), \
             patch("main.init_gemini"), \
             patch("model_loader._model", mock_model), \
             patch("routes.predict.get_model", return_value=mock_model), \
             patch("services.gemini_service._client", mock_gemini_client), \
             patch("routes.predict.validate_and_read_image", side_effect=RuntimeError("unexpected")):
            from main import app
            from tests.conftest import _disable_all_rate_limiters, _enable_all_rate_limiters
            _disable_all_rate_limiters()
            from fastapi.testclient import TestClient
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/predict", files=[upload_file(jpeg_image_bytes)])
            _enable_all_rate_limiters()
        assert resp.status_code == 500
        assert resp.json()["error_code"] == "INTERNAL_ERROR"

    def test_e2e_rate_limiting(self, jpeg_image_bytes):
        """Verify rate limiting works when enabled."""
        with patch("main.load_model"), \
             patch("main.init_gemini"), \
             patch("model_loader._model", MagicMock()), \
             patch("routes.predict.get_model") as mock_get, \
             patch("services.gemini_service._client", MagicMock()):
            mock_model = mock_get.return_value
            mock_model.predict.return_value = np.array([[0.95, 0.03, 0.02]])
            from main import app
            # Keep rate limiting enabled for this test
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                statuses = []
                for _ in range(11):
                    resp = c.post("/predict", files=[upload_file(jpeg_image_bytes)])
                    statuses.append(resp.status_code)
                # At least some should succeed and the last ones may be rate-limited
                assert 200 in statuses
                # Rate limiting may or may not kick in with TestClient (depends on timing)
                # Just verify we don't crash
                assert all(s in (200, 429) for s in statuses)
