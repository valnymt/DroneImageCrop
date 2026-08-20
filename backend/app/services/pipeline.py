from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from app.services.image_encoding import encode_png_data_url
from app.services.opencv_processor import OpenCVProcessor
from app.services.plant_size_analyzer import PlantSizeAnalyzer
from app.services.sam_segmenter import SAMSegmenter
from app.services.schemas import AnalysisResult, PlantSizeStats
from app.services.texture_analyzer import TextureAnalyzer
from app.services.tilt_corrector import TiltCorrector
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
        self.texture = TextureAnalyzer()
        self.tilt = TiltCorrector()
        self.plant_size = PlantSizeAnalyzer()
        self.yield_estimator = YieldEstimator()

    def analyze(
        self,
        path: Path,
        crop: str,
        area_ha: float,
        average_kg: float | None = None,
        enhance: bool = True,
        refine_segmentation: bool = True,
        correct_tilt: bool = True,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> AnalysisResult:
        image = self.cv.load_and_preprocess(path, enhance=enhance)
        # Runs before every other analysis step -- density, coverage, and
        # detection are all measured on whatever geometry `image` ends up
        # with, so a tilt correction has to happen first to actually help,
        # not be computed afterward as a side note.
        tilt = self.tilt.correct(image) if correct_tilt else None
        if tilt is not None and tilt.corrected:
            image = tilt.image
        vegetation = self.cv.vegetation_metrics(image)
        detections = self.detector.detect(image, conf_threshold=conf_threshold)
        plant_count = len(detections)
        # YOLO's boxes are used as SAM prompts, so detection must run first.
        # segment_instances is called once here (not SAMSegmenter.refine,
        # which would re-run the same SAM predictions) so the per-plant
        # masks are available for size/shape stats below instead of only
        # ever being unioned together and discarded.
        instance_masks = self.segmenter.segment_instances(image, detections) if refine_segmentation else None
        refined_mask = (
            self.segmenter.union_masks(instance_masks, vegetation.green_mask.shape[:2])
            if instance_masks is not None
            else vegetation.green_mask
        )
        confidence = 100 * sum(d.confidence for d in detections) / max(plant_count, 1)
        crop_coverage = round(100 * (refined_mask > 0).mean(), 2)
        texture = self.texture.analyze(image, refined_mask)
        # Texture gets a real (not token) weight: two fields at the same
        # color health can still mean very different things -- a uniformly
        # discolored field (drought, nutrient deficiency) keeps a smooth
        # texture, while disease or pest damage tends to look patchy at
        # the same color health. health_score alone can't separate those;
        # texture_pattern (surfaced separately on AnalysisResult) is what
        # actually distinguishes them for the caller.
        health = round(min(100, 0.40 * vegetation.vegetation_score + 0.35 * crop_coverage + 0.25 * texture.uniformity_score), 2)
        per_plant_kg = self.yield_estimator.resolve_per_plant_kg(crop, average_kg)
        estimated = self.yield_estimator.estimate(plant_count, crop, crop_coverage, health, average_kg)
        h, w = image.shape[:2]
        # Same uniform-ground-scale assumption crop_density already makes
        # (area_ha spread evenly across the frame) -- not a new one
        # introduced for this.
        cm2_per_pixel = (area_ha * 1e8) / (w * h)
        plant_size = self.plant_size.analyze(instance_masks, cm2_per_pixel) if instance_masks else None
        return AnalysisResult(
            plant_count=plant_count,
            crop_density=round(plant_count / area_ha, 2),
            crop_coverage=crop_coverage,
            vegetation_score=vegetation.vegetation_score,
            health_score=health,
            texture_uniformity_score=texture.uniformity_score,
            texture_pattern=texture.pattern,
            tilt_corrected=bool(tilt is not None and tilt.corrected),
            tilt_correction_note=tilt.note if tilt is not None else "Perspective correction disabled for this analysis.",
            # Only set when tilt correction actually changed the image's
            # geometry -- detections/image_width/image_height are all
            # relative to the corrected frame at that point, but the
            # caller's own original upload isn't. Omitted otherwise (the
            # common case) since the caller's original upload already
            # matches this pipeline's geometry exactly and re-sending the
            # same bytes would just be wasted payload.
            analyzed_image=encode_png_data_url(image) if (tilt is not None and tilt.corrected) else None,
            plant_size_stats=PlantSizeStats(**asdict(plant_size)) if plant_size is not None else None,
            estimated_yield=estimated,
            average_yield_per_plant_kg=round(per_plant_kg, 3),
            confidence_score=round(confidence, 2),
            detections=detections,
            image_width=w,
            image_height=h,
            segmentation_overlay=encode_png_data_url(_segmentation_overlay(image, refined_mask)),
            heatmap_overlay=encode_png_data_url(vegetation.heatmap),
        )
