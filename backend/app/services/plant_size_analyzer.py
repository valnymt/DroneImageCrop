from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PlantSizeStats:
    plant_count: int
    mean_area_cm2: float
    median_area_cm2: float
    min_area_cm2: float
    max_area_cm2: float
    # Elongation: minAreaRect's long side / short side per plant, averaged.
    # 1.0 = compact/round canopy, higher = narrower/more elongated (e.g.
    # cereal tillers vs. a bushier broadleaf).
    mean_aspect_ratio: float
    # 0-100, derived from the coefficient of variation (std/mean) of
    # per-plant area -- high uniformity means the stand established
    # evenly; low uniformity (a few plants much larger/smaller than the
    # rest) can point at uneven emergence timing, competition, or patchy
    # stress that a single averaged health score wouldn't show at all.
    size_uniformity_score: float


class PlantSizeAnalyzer:
    def analyze(self, instance_masks: list[np.ndarray], cm2_per_pixel: float) -> PlantSizeStats | None:
        """instance_masks: one binary mask per detected plant (see
        SAMSegmenter.segment_instances) -- SAM already computes these as
        part of refining the coverage mask; this only adds up what was
        otherwise being thrown away, not a second segmentation pass.
        cm2_per_pixel comes from the field's own declared area (area_ha)
        spread over the analyzed image's pixel count -- the same uniform-
        ground-scale assumption crop_density already makes elsewhere in
        this pipeline, not a new one introduced here.
        """
        areas_cm2: list[float] = []
        aspect_ratios: list[float] = []
        for mask in instance_masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(largest)
            if area_px <= 0:
                continue
            areas_cm2.append(area_px * cm2_per_pixel)
            (_, _), (side_a, side_b), _ = cv2.minAreaRect(largest)
            short_side, long_side = sorted((side_a, side_b))
            aspect_ratios.append(long_side / short_side if short_side > 1e-6 else 1.0)

        if not areas_cm2:
            return None

        areas = np.array(areas_cm2)
        mean_area = float(areas.mean())
        coefficient_of_variation = float(areas.std()) / mean_area if mean_area > 1e-9 else 0.0
        uniformity = max(0.0, 100 * (1 - min(coefficient_of_variation, 1.0)))

        return PlantSizeStats(
            plant_count=len(areas_cm2),
            mean_area_cm2=round(mean_area, 2),
            median_area_cm2=round(float(np.median(areas)), 2),
            min_area_cm2=round(float(areas.min()), 2),
            max_area_cm2=round(float(areas.max()), 2),
            mean_aspect_ratio=round(float(np.mean(aspect_ratios)), 2),
            size_uniformity_score=round(uniformity, 2),
        )
