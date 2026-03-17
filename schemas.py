from enum import Enum

from pydantic import BaseModel, Field
from typing import List, Optional


class PlantType(str, Enum):
    tomato = "tomato"
    corn = "corn"
    pepper = "pepper"
    apple = "apple"
    strawberry = "strawberry"


class AllPlantType(str, Enum):
    potato = "potato"
    tomato = "tomato"
    corn = "corn"
    pepper = "pepper"
    apple = "apple"
    strawberry = "strawberry"


class PredictionResponse(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float
    confidence_percent: str

    model_config = {"populate_by_name": True}


class AdviceRequest(BaseModel):
    disease: str
    confidence: float = Field(..., ge=0, le=1)
    plant_type: Optional[AllPlantType] = None


class AdviceResponse(BaseModel):
    disease: str
    treatment: str
    prevention: str
    care_instructions: str


class ActionItem(BaseModel):
    week: str
    action: str
    details: str


class CropCalendarRequest(BaseModel):
    disease: str
    confidence: float = Field(..., ge=0, le=1)
    season: Optional[str] = None
    location: Optional[str] = None
    plant_type: Optional[AllPlantType] = None


class CropCalendarResponse(BaseModel):
    disease: str
    season: str
    timeline: List[ActionItem]


class SeverityResponse(BaseModel):
    severity_level: str
    affected_area_percent: float
    urgency: str
    immediate_actions: List[str]
    detailed_analysis: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gemini_available: bool


class AnalyzeResponse(BaseModel):
    plant_type: str
    disease_name: str
    is_healthy: bool
    confidence: float = Field(..., ge=0, le=1)
    description: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
