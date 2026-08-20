from pathlib import Path

import cv2
import numpy as np

from app.services.image_encoding import encode_png_data_url
from app.services.opencv_processor import OpenCVProcessor
from app.services.sam_segmenter import SAMSegmenter
from app.services.schemas import AnalysisResult
from app.services.yield_estimator import YieldEstimator
from app.services.yolo_detector import CONF_THRESHOLD, YOLODetector

# Magenta rather than green -- a green tint blended onto already-green
# vegetation is nearly invisible; magenta doesn't occur naturally in field
# photos, so the segmented region actually stands out.
SEGMENTATION_TINT_BGR = (230, 40, 220)
SEGMENTATION_TINT_ALPHA = 0.5


def _segmentation_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The analyzed image with a translucent tint over segmented (mask>0)
    regions -- lets the Segmentation view show exactly what was measured,
    rather than a bare black-and-white mask."""
    tint = np.full_like(image, SEGMENTATION_TINT_BGR)
    blended = cv2.addWeighted(image, 1 - SEGMENTATION_TINT_ALPHA, tint, SEGMENTATION_TINT_ALPHA, 0)
    overlay = image.copy()
    overlay[mask > 0] = blended[mask > 0]
    return overlay


class CropAnalysisPipeline:
    def __init__(self) -> None:
        self.cv = OpenCVProcessor()
        self.detector = YOLODetector()
        self.segmenter = SAMSegmenter()
        self.yield_estimator = YieldEstimator()

    def analyze(
        self,
        path: Path,
        crop: str,
        area_ha: float,
        average_kg: float | None = None,
        enhance: bool = True,
        refine_segmentation: bool = True,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> AnalysisResult:
        image = self.cv.load_and_preprocess(path, enhance=enhance)
        vegetation = self.cv.vegetation_metrics(image)
        detections = self.detector.detect(image, conf_threshold=conf_threshold)
        plant_count = len(detections)
        # YOLO's boxes are used as SAM prompts, so detection must run first.
        refined_mask = (
            self.segmenter.refine(image, vegetation.green_mask, detections)
            if refine_segmentation
            else vegetation.green_mask
        )
        confidence = 100 * sum(d.confidence for d in detections) / max(plant_count, 1)
        per_plant_kg = self.yield_estimator.resolve_per_plant_kg(crop, average_kg)
        estimated = self.yield_estimator.estimate(
            plant_count, crop, vegetation.coverage_percent, vegetation.health_score, average_kg
        )
        h, w = image.shape[:2]
        return AnalysisResult(
            plant_count=plant_count,
            crop_density=round(plant_count / area_ha, 2),
            crop_coverage=round(100 * (refined_mask > 0).mean(), 2),
            vegetation_score=vegetation.vegetation_score,
            health_score=vegetation.health_score,
            estimated_yield=estimated,
            average_yield_per_plant_kg=round(per_plant_kg, 3),
            confidence_score=round(confidence, 2),
            detections=detections,
            image_width=w,
            image_height=h,
            segmentation_overlay=encode_png_data_url(_segmentation_overlay(image, refined_mask)),
            heatmap_overlay=encode_png_data_url(vegetation.heatmap),
        )
