from pydantic import BaseModel, Field


class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str


class AnalysisResult(BaseModel):
    plant_count: int = Field(ge=0)
    crop_density: float = Field(ge=0)
    crop_coverage: float = Field(ge=0, le=100)
    vegetation_score: float = Field(ge=0, le=100)
    health_score: float = Field(ge=0, le=100)
    estimated_yield: float = Field(ge=0)
    confidence_score: float = Field(ge=0, le=100)
    detections: list[Detection]
