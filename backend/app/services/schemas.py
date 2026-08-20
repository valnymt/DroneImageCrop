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
    # Pixel space that `detections` coordinates are relative to -- the
    # preprocessed/resized image the pipeline actually analyzed, NOT
    # necessarily the original upload's raw dimensions (load_and_preprocess
    # caps at 1600px). The frontend needs this to scale boxes correctly
    # onto the displayed (original) photo.
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    # data: URLs (base64 PNG), ready to drop straight into an <img src>.
    segmentation_overlay: str
    heatmap_overlay: str


class InspectResult(BaseModel):
    """Best-effort, low-confidence suggestions for pre-filling the analyze
    form -- never authoritative. crop_type comes from zero-shot CLIP
    classification (see app/services/crop_classifier.py; ~69% accurate,
    see backend/training/CLIP_CLASSIFIER_EVAL_REPORT.md for exactly where
    it's wrong). estimated_area_hectares is null whenever the photo has no
    altitude metadata to derive it from -- that is the common case, not an
    error."""

    crop_type: str
    confidence: float = Field(ge=0, le=100)
    estimated_area_hectares: float | None = Field(default=None, ge=0)
    area_source: str
