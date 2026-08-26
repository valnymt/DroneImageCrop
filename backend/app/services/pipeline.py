from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from app.services.image_encoding import encode_png_data_url
from app.services.open_vocab_detector import OpenVocabDetector
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

# Empirically confirmed (see HANDOFF.md's Phase O): the fine-tuned YOLO
# checkpoint returns zero detections -- not borderline misses, genuinely
# nothing -- on photos stylistically unlike its ~823-image training set,
# even when real vegetation is clearly visible. Below this coverage,
# "zero detections" is far more likely a genuinely bare-soil photo than a
# domain-mismatch failure, so the (much slower) fallback only runs when
# there's real vegetation the primary detector plausibly missed.
FALLBACK_COVERAGE_THRESHOLD = 2.0

# Approximate standing-plant densities used only when a field has clear
# vegetation but neither object detector can resolve individual plants. The
# coverage factor keeps this conservative and avoids reporting zero for a
# dense field photo whose plants are below detector resolution.
VEGETATION_DENSITY_PER_HA = {
    "wheat": 180_000,
    "rice": 220_000,
    "corn": 65_000,
    "soybean": 250_000,
    "tomato": 20_000,
    "_default": 100_000,
}


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
        self.fallback_detector = OpenVocabDetector()
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
        estimated_plant_count: int | None = None
        detection_method = "fine_tuned"
        detection_note = f"{len(detections)} plant(s) detected by the fine-tuned model."
        # A single box on a visibly dense crop is usually an under-count, not
        # a one-plant field. Let the general detector supplement the fine-tuned
        # model in that case, then keep whichever result finds more plants.
        if len(detections) < 3 and vegetation.coverage_percent >= FALLBACK_COVERAGE_THRESHOLD:
            primary_count = len(detections)
            fallback_detections = self.fallback_detector.detect(image)
            if len(fallback_detections) > len(detections):
                detections = fallback_detections
                detection_method = "general_fallback"
                detection_note = (
                    f"The fine-tuned model found only {primary_count} plant(s) despite {vegetation.coverage_percent:.0f}% visible "
                    "vegetation coverage (this photo may look unlike its training images) -- fell back to a "
                    f"general-purpose zero-shot detector, which found {len(detections)} plant(s). Less precise "
                    "than the fine-tuned model on photos it was actually trained for."
                )
            else:
                detection_note = (
                    f"No plants detected by either the fine-tuned model or the general-purpose fallback, "
                    f"despite {vegetation.coverage_percent:.0f}% visible vegetation coverage."
                )
        if len(detections) < 3 and vegetation.coverage_percent >= FALLBACK_COVERAGE_THRESHOLD:
            # Individual plants can be sub-pixel/small in a wide field view.
            # Keep the detector result empty for overlays, but provide a
            # conservative population estimate for density and yield rather
            # than returning the unusable 0-plant/0-yield pair.
            density_per_ha = VEGETATION_DENSITY_PER_HA.get(crop.lower(), VEGETATION_DENSITY_PER_HA["_default"])
            estimated_count = max(1, round(area_ha * density_per_ha * (vegetation.coverage_percent / 100) * 0.25))
            # Keep plant_count strictly as the number of boxes actually
            # detected. The area/density calculation is a separate planning
            # estimate and must never masquerade as an exact image count.
            estimated_plant_count = estimated_count
            detection_note = detection_note + (
                f" Individual plants were too small to box reliably; population estimated from "
                f"{vegetation.coverage_percent:.0f}% visible vegetation coverage at approximately "
                f"{estimated_plant_count:,} plants. {len(detections):,} plant(s) were directly detected."
            )
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
        mask_confidence = getattr(vegetation, "mask_confidence_score", 0.0)
        if estimated_plant_count is not None:
            # There are no reliable individual boxes, so report confidence
            # in the RGB crop mask with a conservative penalty for estimating
            # population rather than pretending greenness is detector
            # confidence.
            confidence = 0.75 * mask_confidence
        else:
            detector_confidence = 100 * sum(d.confidence for d in detections) / max(plant_count, 1)
            # Detection confidence and RGB-mask confidence describe different
            # evidence sources. Keep both in the combined reliability score;
            # vegetation_score is deliberately not used here.
            confidence = 0.75 * detector_confidence + 0.25 * mask_confidence
        crop_coverage = round(100 * (refined_mask > 0).mean(), 2)
        texture = self.texture.analyze(image, refined_mask)
        # Health is based on the crop's greenness and visible crop coverage.
        # Texture is deliberately not part of the health score: row gaps,
        # planting geometry, shadows, and field perspective can all look
        # "patchy" without meaning that the crop is unhealthy. Texture is
        # surfaced separately as a diagnostic pattern and only informs the
        # recommendation when the color/coverage health is already reduced.
        health = round(min(100, 0.65 * vegetation.vegetation_score + 0.35 * crop_coverage), 2)
        per_plant_kg = self.yield_estimator.resolve_per_plant_kg(crop, average_kg)
        h, w = image.shape[:2]
        # Same uniform-ground-scale assumption crop_density already makes
        # (area_ha spread evenly across the frame) -- not a new one
        # introduced for this.
        population_count = estimated_plant_count if estimated_plant_count is not None else plant_count
        cm2_per_pixel = (area_ha * 1e8) / (w * h)
        plant_size = self.plant_size.analyze(instance_masks, cm2_per_pixel) if instance_masks else None
        yield_size_factor, yield_size_note = self.yield_estimator.size_adjustment(plant_size, population_count, area_ha)
        estimated = self.yield_estimator.estimate(
            population_count, crop, crop_coverage, health, average_kg, plant_size, area_ha
        )
        return AnalysisResult(
            plant_count=plant_count,
            estimated_plant_count=estimated_plant_count,
            # Density/yield use the estimated population when direct boxes
            # are insufficient, while plant_count remains the direct count.
            crop_density=round(population_count / area_ha, 2),
            crop_coverage=crop_coverage,
            vegetation_score=vegetation.vegetation_score,
            health_score=health,
            texture_uniformity_score=texture.uniformity_score,
            texture_pattern=texture.pattern,
            detection_method=detection_method,
            detection_note=detection_note,
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
            yield_size_factor=yield_size_factor,
            yield_size_note=yield_size_note,
            confidence_score=round(confidence, 2),
            detections=detections,
            image_width=w,
            image_height=h,
            segmentation_overlay=encode_png_data_url(_segmentation_overlay(image, refined_mask)),
            heatmap_overlay=encode_png_data_url(vegetation.heatmap),
        )
