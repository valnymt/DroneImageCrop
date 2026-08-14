from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.crop_classifier import CropClassifier
from app.services.field_area_estimator import estimate_area_hectares
from app.services.pipeline import CropAnalysisPipeline
from app.services.schemas import AnalysisResult, InspectResult

router = APIRouter(tags=["analysis"])
pipeline = CropAnalysisPipeline()
classifier = CropClassifier()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze(
    image: UploadFile = File(...),
    crop_type: str = Form(...),
    field_size_hectares: float = Form(..., gt=0),
    average_yield_per_plant_kg: float = Form(0.5, gt=0),
) -> AnalysisResult:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only JPG and PNG images are supported.")
    suffix = Path(image.filename or "field.jpg").suffix
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        path = Path(tmp.name)
    try:
        return pipeline.analyze(
            path, crop_type, field_size_hectares, average_yield_per_plant_kg
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@router.post("/inspect", response_model=InspectResult)
async def inspect(image: UploadFile = File(...)) -> InspectResult:
    """Best-effort pre-fill suggestions for the analyze form -- run on
    upload, before the user has entered anything. Never blocks or replaces
    /analyze; a failure here should never prevent a real analysis."""
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only JPG and PNG images are supported.")
    suffix = Path(image.filename or "field.jpg").suffix
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        path = Path(tmp.name)
    try:
        cv_image = cv2.imread(str(path))
        if cv_image is None:
            raise HTTPException(422, "The uploaded image could not be decoded.")
        crop_type, confidence = classifier.classify(cv_image)
        area_ha, area_source = estimate_area_hectares(path)
        return InspectResult(
            crop_type=crop_type,
            confidence=round(confidence * 100, 2),
            estimated_area_hectares=area_ha,
            area_source=area_source,
        )
    finally:
        path.unlink(missing_ok=True)
