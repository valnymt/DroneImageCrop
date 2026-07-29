from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VegetationMetrics:
    green_mask: np.ndarray
    coverage_percent: float
    vegetation_score: float
    health_score: float


class OpenCVProcessor:
    def load_and_preprocess(self, path: Path, max_side: int = 1600) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("The uploaded image could not be decoded.")
        h, w = image.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Bilateral filtering suppresses drone sensor noise while retaining leaf edges.
        denoised = cv2.bilateralFilter(image, 7, 45, 45)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE restores local contrast without overexposing bright soil regions.
        l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def vegetation_metrics(self, image: np.ndarray) -> VegetationMetrics:
        b, g, r = cv2.split(image.astype(np.float32))
        # Excess Green (2G-R-B) is a robust RGB-only vegetation proxy.
        exg = 2 * g - r - b
        normalized = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        coverage = 100 * cv2.countNonZero(mask) / mask.size
        green_dominance = float(np.mean(np.clip(exg[mask > 0], 0, 255))) if np.any(mask) else 0
        vegetation = min(100.0, green_dominance / 1.5)
        health = min(100.0, 0.55 * vegetation + 0.45 * coverage)
        return VegetationMetrics(mask, round(coverage, 2), round(vegetation, 2), round(health, 2))
