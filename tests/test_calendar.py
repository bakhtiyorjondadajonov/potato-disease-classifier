import json
from unittest.mock import patch

from fastapi import HTTPException


VALID_JSON_1 = json.dumps([{"week": "Week 1", "action": "Spray", "details": "Apply fungicide"}])
VALID_JSON_4 = json.dumps([
    {"week": "Week 1", "action": "A1", "details": "D1"},
    {"week": "Week 2", "action": "A2", "details": "D2"},
    {"week": "Week 3", "action": "A3", "details": "D3"},
    {"week": "Week 4", "action": "A4", "details": "D4"},
])


class TestCalendarSuccess:
    def test_calendar_success_valid_json(self, client):
        with patch("routes.calendar.generate_text", return_value=VALID_JSON_1):
            resp = client.post("/crop-calendar", json={"disease": "Early Blight", "confidence": 0.9})
        assert resp.status_code == 200
        assert resp.json()["timeline"][0]["week"] == "Week 1"

    def test_calendar_success_multiple_items(self, client):
        with patch("routes.calendar.generate_text", return_value=VALID_JSON_4):
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.5})
        assert len(resp.json()["timeline"]) == 4

    def test_calendar_json_in_markdown(self, client):
        md = f"```json\n{VALID_JSON_1}\n```"
        with patch("routes.calendar.generate_text", return_value=md):
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 200
        assert resp.json()["timeline"][0]["action"] == "Spray"


class TestCalendarFallback:
    def test_calendar_invalid_json_fallback(self, client):
        with patch("routes.calendar.generate_text", return_value="Not JSON at all"):
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 200
        assert len(resp.json()["timeline"]) == 3

    def test_calendar_partial_json_fallback(self, client):
        with patch("routes.calendar.generate_text", return_value="[{broken"):
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.5})
        assert len(resp.json()["timeline"]) == 3

    def test_calendar_fallback_includes_disease(self, client):
        with patch("routes.calendar.generate_text", return_value="garbage"):
            resp = client.post("/crop-calendar", json={"disease": "Late Blight", "confidence": 0.5})
        details = " ".join(item["details"] for item in resp.json()["timeline"])
        assert "Late Blight" in details

    def test_calendar_fallback_uses_plant_type(self, client):
        with patch("routes.calendar.generate_text", return_value="garbage"):
            resp = client.post("/crop-calendar", json={
                "disease": "Rust", "confidence": 0.5, "plant_type": "tomato",
            })
        details = " ".join(item["details"] for item in resp.json()["timeline"])
        assert "tomato" in details


class TestCalendarPlantType:
    def test_calendar_plant_type_none_defaults_potato(self, client):
        with patch("routes.calendar.generate_text", return_value=VALID_JSON_1) as mock_gen:
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.5})
        assert "potato" in mock_gen.call_args[0][0].lower()

    def test_calendar_with_plant_type_tomato(self, client):
        with patch("routes.calendar.generate_text", return_value=VALID_JSON_1) as mock_gen:
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.5, "plant_type": "tomato"})
        assert "tomato" in mock_gen.call_args[0][0].lower()

    def test_calendar_with_season_and_location(self, client):
        with patch("routes.calendar.generate_text", return_value=VALID_JSON_1) as mock_gen:
            resp = client.post("/crop-calendar", json={
                "disease": "X", "confidence": 0.5, "season": "summer", "location": "Poland"
            })
        prompt = mock_gen.call_args[0][0].lower()
        assert "summer" in prompt
        assert "poland" in prompt


class TestCalendarErrors:
    def test_calendar_gemini_503(self, client):
        with patch("routes.calendar.generate_text", side_effect=HTTPException(503, "unavailable")):
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 503

    def test_calendar_gemini_generic_exception(self, client):
        with patch("routes.calendar.generate_text", side_effect=RuntimeError("boom")):
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 502

    def test_calendar_confidence_zero(self, client):
        with patch("routes.calendar.generate_text", return_value=VALID_JSON_1):
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 0.0})
        assert resp.status_code == 200

    def test_calendar_confidence_one(self, client):
        with patch("routes.calendar.generate_text", return_value=VALID_JSON_1):
            resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 1.0})
        assert resp.status_code == 200

    def test_calendar_confidence_invalid(self, client):
        resp = client.post("/crop-calendar", json={"disease": "X", "confidence": 1.5})
        assert resp.status_code == 422
