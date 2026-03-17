import pytest
from pydantic import ValidationError

from schemas import (
    PlantType,
    AllPlantType,
    PredictionResponse,
    AdviceRequest,
    AdviceResponse,
    CropCalendarRequest,
    ActionItem,
    CropCalendarResponse,
    SeverityResponse,
    HealthResponse,
    AnalyzeResponse,
    ErrorResponse,
)


class TestPlantType:
    def test_plant_type_valid_values(self):
        for val in ("tomato", "corn", "pepper", "apple", "strawberry"):
            assert PlantType(val).value == val

    def test_plant_type_invalid_value(self):
        with pytest.raises(ValueError):
            PlantType("banana")

    def test_plant_type_values_list(self):
        assert set(e.value for e in PlantType) == {"tomato", "corn", "pepper", "apple", "strawberry"}


class TestAllPlantType:
    def test_all_plant_type_includes_potato(self):
        assert AllPlantType("potato").value == "potato"

    def test_all_plant_type_valid_values(self):
        for val in ("potato", "tomato", "corn", "pepper", "apple", "strawberry"):
            assert AllPlantType(val).value == val

    def test_all_plant_type_invalid_value(self):
        with pytest.raises(ValueError):
            AllPlantType("banana")


class TestPredictionResponse:
    def test_prediction_response_alias(self):
        resp = PredictionResponse(**{"class": "Early Blight", "confidence": 0.95, "confidence_percent": "95.0%"})
        assert resp.class_name == "Early Blight"

    def test_prediction_response_by_field_name(self):
        resp = PredictionResponse(class_name="Healthy", confidence=0.99, confidence_percent="99.0%")
        assert resp.class_name == "Healthy"


class TestAdviceRequest:
    def test_advice_request_confidence_boundaries(self):
        AdviceRequest(disease="X", confidence=0.0)
        AdviceRequest(disease="X", confidence=1.0)

    def test_advice_request_confidence_too_low(self):
        with pytest.raises(ValidationError):
            AdviceRequest(disease="X", confidence=-0.1)

    def test_advice_request_confidence_too_high(self):
        with pytest.raises(ValidationError):
            AdviceRequest(disease="X", confidence=1.1)

    def test_advice_request_plant_type_optional(self):
        req = AdviceRequest(disease="X", confidence=0.5)
        assert req.plant_type is None

    def test_advice_request_plant_type_valid(self):
        req = AdviceRequest(disease="X", confidence=0.5, plant_type="tomato")
        assert req.plant_type == AllPlantType.tomato

    def test_advice_request_plant_type_invalid(self):
        with pytest.raises(ValidationError):
            AdviceRequest(disease="X", confidence=0.5, plant_type="banana")


class TestAdviceResponse:
    def test_advice_response_construction(self):
        resp = AdviceResponse(
            disease="Early Blight",
            treatment="Apply fungicide",
            prevention="Rotate crops",
            care_instructions="Water at base",
        )
        assert resp.disease == "Early Blight"
        assert resp.treatment == "Apply fungicide"
        assert resp.prevention == "Rotate crops"
        assert resp.care_instructions == "Water at base"


class TestCropCalendarRequest:
    def test_crop_calendar_request_optional_fields(self):
        req = CropCalendarRequest(disease="X", confidence=0.5)
        assert req.season is None
        assert req.location is None
        assert req.plant_type is None

    def test_crop_calendar_request_confidence_bounds(self):
        CropCalendarRequest(disease="X", confidence=0.0)
        CropCalendarRequest(disease="X", confidence=1.0)
        with pytest.raises(ValidationError):
            CropCalendarRequest(disease="X", confidence=-0.1)
        with pytest.raises(ValidationError):
            CropCalendarRequest(disease="X", confidence=1.1)


class TestActionItem:
    def test_action_item_construction(self):
        item = ActionItem(week="Week 1", action="Spray", details="Apply fungicide")
        assert item.week == "Week 1"
        assert item.action == "Spray"
        assert item.details == "Apply fungicide"


class TestCropCalendarResponse:
    def test_crop_calendar_response_with_timeline(self):
        items = [ActionItem(week="Week 1", action="A", details="D")]
        resp = CropCalendarResponse(disease="X", season="summer", timeline=items)
        assert len(resp.timeline) == 1
        assert resp.timeline[0].week == "Week 1"


class TestSeverityResponse:
    def test_severity_response_construction(self):
        resp = SeverityResponse(
            severity_level="mild",
            affected_area_percent=10.0,
            urgency="low",
            immediate_actions=["action1"],
            detailed_analysis="test",
        )
        assert resp.severity_level == "mild"
        assert resp.affected_area_percent == 10.0
        assert resp.urgency == "low"
        assert resp.immediate_actions == ["action1"]
        assert resp.detailed_analysis == "test"


class TestHealthResponse:
    def test_health_response_construction(self):
        resp = HealthResponse(status="ok", model_loaded=True, gemini_available=True)
        assert resp.status == "ok"
        assert resp.model_loaded is True
        assert resp.gemini_available is True


class TestAnalyzeResponse:
    def test_analyze_response_confidence_bounds(self):
        AnalyzeResponse(plant_type="tomato", disease_name="X", is_healthy=False, confidence=0.0, description="d")
        AnalyzeResponse(plant_type="tomato", disease_name="X", is_healthy=False, confidence=1.0, description="d")
        with pytest.raises(ValidationError):
            AnalyzeResponse(plant_type="tomato", disease_name="X", is_healthy=False, confidence=-0.1, description="d")
        with pytest.raises(ValidationError):
            AnalyzeResponse(plant_type="tomato", disease_name="X", is_healthy=False, confidence=1.1, description="d")

    def test_analyze_response_all_fields(self):
        resp = AnalyzeResponse(plant_type="corn", disease_name="Rust", is_healthy=False, confidence=0.8, description="desc")
        assert resp.plant_type == "corn"
        assert resp.disease_name == "Rust"
        assert resp.is_healthy is False
        assert resp.confidence == 0.8
        assert resp.description == "desc"


class TestErrorResponse:
    def test_error_response_optional_error_code(self):
        resp = ErrorResponse(detail="something failed")
        assert resp.error_code is None
        resp2 = ErrorResponse(detail="fail", error_code="ERR_001")
        assert resp2.error_code == "ERR_001"
