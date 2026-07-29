from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.pipeline import CropAnalysisPipeline
from app.services.schemas import AnalysisResult

router = APIRouter(tags=["analysis"])
pipeline = CropAnalysisPipeline()
history: list[AnalysisResult] = []


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
        result = pipeline.analyze(
            path, crop_type, field_size_hectares, average_yield_per_plant_kg
        )
        history.insert(0, result)
        return result
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@router.get("/history", response_model=list[AnalysisResult])
def get_history() -> list[AnalysisResult]:
    return history
