from unittest.mock import patch

from fastapi import HTTPException


FULL_RESPONSE = """TREATMENT:
Apply copper-based fungicide immediately. Repeat every 7-10 days.

PREVENTION:
Rotate crops annually. Ensure proper spacing.

CARE INSTRUCTIONS:
Water at the base. Remove affected leaves promptly."""


class TestAdviceSuccess:
    def test_advice_success_full_sections(self, client):
        with patch("routes.advice.generate_text", return_value=FULL_RESPONSE):
            resp = client.post("/advice", json={"disease": "Early Blight", "confidence": 0.9})
        assert resp.status_code == 200
        data = resp.json()
        assert "fungicide" in data["treatment"].lower()
        assert "rotate" in data["prevention"].lower()
        assert "water" in data["care_instructions"].lower()

    def test_advice_with_plant_type_tomato(self, client):
        with patch("routes.advice.generate_text", return_value=FULL_RESPONSE) as mock_gen:
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5, "plant_type": "tomato"})
        assert resp.status_code == 200
        assert "tomato" in mock_gen.call_args[0][0].lower()

    def test_advice_plant_type_none_defaults_potato(self, client):
        with patch("routes.advice.generate_text", return_value=FULL_RESPONSE) as mock_gen:
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 200
        assert "potato" in mock_gen.call_args[0][0].lower()


class TestAdviceParsing:
    def test_advice_missing_treatment_section(self, client):
        text = "PREVENTION:\nDo X.\n\nCARE INSTRUCTIONS:\nDo Y."
        with patch("routes.advice.generate_text", return_value=text):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 200
        assert resp.json()["treatment"] == "No information available."

    def test_advice_missing_prevention_section(self, client):
        text = "TREATMENT:\nDo A.\n\nCARE INSTRUCTIONS:\nDo B."
        with patch("routes.advice.generate_text", return_value=text):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert resp.json()["prevention"] == "No information available."

    def test_advice_missing_care_section(self, client):
        text = "TREATMENT:\nDo A.\n\nPREVENTION:\nDo B."
        with patch("routes.advice.generate_text", return_value=text):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert resp.json()["care_instructions"] == "No information available."

    def test_advice_empty_response(self, client):
        with patch("routes.advice.generate_text", return_value=""):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        data = resp.json()
        assert data["treatment"] == "No information available."
        assert data["prevention"] == "No information available."
        assert data["care_instructions"] == "No information available."

    def test_advice_care_instructions_underscore(self, client):
        text = "TREATMENT:\nA.\n\nPREVENTION:\nB.\n\nCARE_INSTRUCTIONS:\nC stuff."
        with patch("routes.advice.generate_text", return_value=text):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert "C stuff" in resp.json()["care_instructions"]

    def test_advice_multiline_sections(self, client):
        text = "TREATMENT:\nLine1.\nLine2.\n\nPREVENTION:\nP1.\n\nCARE INSTRUCTIONS:\nC1."
        with patch("routes.advice.generate_text", return_value=text):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert "Line1." in resp.json()["treatment"]
        assert "Line2." in resp.json()["treatment"]


class TestAdviceErrors:
    def test_advice_gemini_503(self, client):
        with patch("routes.advice.generate_text", side_effect=HTTPException(503, "unavailable")):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 503

    def test_advice_gemini_generic_exception(self, client):
        with patch("routes.advice.generate_text", side_effect=RuntimeError("boom")):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.5})
        assert resp.status_code == 502

    def test_advice_confidence_zero(self, client):
        with patch("routes.advice.generate_text", return_value=FULL_RESPONSE):
            resp = client.post("/advice", json={"disease": "X", "confidence": 0.0})
        assert resp.status_code == 200

    def test_advice_confidence_one(self, client):
        with patch("routes.advice.generate_text", return_value=FULL_RESPONSE):
            resp = client.post("/advice", json={"disease": "X", "confidence": 1.0})
        assert resp.status_code == 200

    def test_advice_confidence_negative(self, client):
        resp = client.post("/advice", json={"disease": "X", "confidence": -0.1})
        assert resp.status_code == 422

    def test_advice_confidence_over_one(self, client):
        resp = client.post("/advice", json={"disease": "X", "confidence": 1.1})
        assert resp.status_code == 422

    def test_advice_missing_disease_field(self, client):
        resp = client.post("/advice", json={"confidence": 0.5})
        assert resp.status_code == 422
