import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from pydantic import ValidationError

from app.services.crop_classifier import CropClassifier
from app.services.field_area_estimator import estimate_area_hectares
from app.services.pipeline import CropAnalysisPipeline
from app.services.report_generator import generate_report_pdf
from app.services.schemas import AnalysisResult, InspectResult

router = APIRouter(tags=["analysis"])
pipeline = CropAnalysisPipeline()
classifier = CropClassifier()


def _safe_pdf_filename(field_name: str, analysis_date: str) -> str:
    # field_name/analysis_date are free-text user input that ends up in an
    # HTTP response header -- restrict to a safe character set rather than
    # passing them through directly, so nothing can inject header syntax.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{field_name}-{analysis_date}").strip("-")
    return f"{slug or 'agrisight-report'}.pdf"


@router.post("/analyze", response_model=AnalysisResult)
async def analyze(
    image: UploadFile = File(...),
    crop_type: str = Form(...),
    field_size_hectares: float = Form(..., gt=0),
    average_yield_per_plant_kg: float = Form(0.5, gt=0),
    enhance: bool = Form(True),
    refine_segmentation: bool = Form(True),
    conf_threshold: float = Form(0.25, ge=0, le=1),
) -> AnalysisResult:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only JPG and PNG images are supported.")
    suffix = Path(image.filename or "field.jpg").suffix
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        path = Path(tmp.name)
    try:
        return pipeline.analyze(
            path, crop_type, field_size_hectares, average_yield_per_plant_kg,
            enhance=enhance, refine_segmentation=refine_segmentation, conf_threshold=conf_threshold,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@router.post("/inspect", response_model=InspectResult)
async def inspect(
    image: UploadFile = File(...),
    manual_altitude_m: float | None = Form(None, gt=0),
) -> InspectResult:
    """Best-effort pre-fill suggestions for the analyze form -- run on
    upload, before the user has entered anything. Never blocks or replaces
    /analyze; a failure here should never prevent a real analysis.

    manual_altitude_m is the one last-resort field the frontend surfaces
    only when area_source comes back "unavailable" on a first (no-altitude)
    call -- it is never asked for up front.
    """
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
        area_ha, area_source = estimate_area_hectares(path, crop_type, manual_altitude_m)
        return InspectResult(
            crop_type=crop_type,
            confidence=round(confidence * 100, 2),
            estimated_area_hectares=area_ha,
            area_source=area_source,
        )
    finally:
        path.unlink(missing_ok=True)


@router.post("/report")
async def report(
    image: UploadFile = File(...),
    result: str = Form(...),
    field_name: str = Form(...),
    crop_type: str = Form(...),
    field_area_hectares: float = Form(..., gt=0),
    analysis_date: str = Form(...),
    health_label: str = Form(...),
    health_copy: str = Form(...),
    recommendation: str = Form(...),
) -> Response:
    """Regenerates a PDF from an already-completed analysis -- takes the
    same image plus the AnalysisResult the frontend already has, rather
    than re-running the CV pipeline."""
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only JPG and PNG images are supported.")
    try:
        analysis_result = AnalysisResult.model_validate_json(result)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    image_bytes = await image.read()
    np_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if np_image is None:
        raise HTTPException(422, "The uploaded image could not be decoded.")

    pdf_bytes = generate_report_pdf(
        np_image, analysis_result, field_name, crop_type, field_area_hectares,
        analysis_date, health_label, health_copy, recommendation,
    )
    filename = _safe_pdf_filename(field_name, analysis_date)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
