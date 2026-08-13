from pathlib import Path

from app.services.opencv_processor import OpenCVProcessor
from app.services.sam_segmenter import SAMSegmenter
from app.services.schemas import AnalysisResult
from app.services.yield_estimator import YieldEstimator
from app.services.yolo_detector import YOLODetector


class CropAnalysisPipeline:
    def __init__(self) -> None:
        self.cv = OpenCVProcessor()
        self.detector = YOLODetector()
        self.segmenter = SAMSegmenter()
        self.yield_estimator = YieldEstimator()

    def analyze(self, path: Path, crop: str, area_ha: float, average_kg: float) -> AnalysisResult:
        image = self.cv.load_and_preprocess(path)
        vegetation = self.cv.vegetation_metrics(image)
        detections = self.detector.detect(image)
        plant_count = len(detections)
        # YOLO's boxes are used as SAM prompts, so detection must run first.
        refined_mask = self.segmenter.refine(image, vegetation.green_mask, detections)
        confidence = 100 * sum(d.confidence for d in detections) / max(plant_count, 1)
        estimated = self.yield_estimator.estimate(
            plant_count, crop, average_kg, vegetation.coverage_percent, vegetation.health_score
        )
        return AnalysisResult(
            plant_count=plant_count,
            crop_density=round(plant_count / area_ha, 2),
            crop_coverage=round(100 * (refined_mask > 0).mean(), 2),
            vegetation_score=vegetation.vegetation_score,
            health_score=vegetation.health_score,
            estimated_yield=estimated,
            confidence_score=round(confidence, 2),
            detections=detections,
        )
