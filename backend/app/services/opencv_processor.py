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
    vari_score: float = 0.0
    exgr_score: float = 0.0


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

        # Excess Green (2G-R-B): general-purpose vegetation proxy.
        exg = 2 * g - r - b
        # VARI (G-R)/(G+R-B): roughly illumination-invariant; unlike ExG it
        # divides out brightness, so it doesn't get fooled by bright, dry,
        # yellow-brown residue that happens to still be "greener than blue".
        denom = g + r - b
        vari = np.divide(g - r, denom, out=np.zeros_like(denom), where=np.abs(denom) > 1e-6)
        # ExGR = ExG - ExR (ExR = 1.4R - G): the extra -1.4*R term punishes
        # red-leaning soil/residue harder than ExG alone does.
        exgr = exg - 1.4 * r + g

        # Each index makes different mistakes (ExG over-triggers on yellow
        # soil, VARI is noisy in shadow, ExGR is sensitive to saturation),
        # so requiring at least 2-of-3 to agree a pixel is vegetation
        # rejects the false positives any single index would let through.
        exg_mask = self._otsu_mask(exg)
        vari_mask = self._otsu_mask(vari)
        exgr_mask = self._otsu_mask(exgr)
        votes = (exg_mask > 0).astype(np.uint8) + (vari_mask > 0).astype(np.uint8) + (exgr_mask > 0).astype(np.uint8)
        mask = np.where(votes >= 2, 255, 0).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        coverage = 100 * cv2.countNonZero(mask) / mask.size
        in_mask = mask > 0
        green_dominance = float(np.mean(np.clip(exg[in_mask], 0, 255))) if np.any(mask) else 0.0
        vari_dominance = float(np.mean(np.clip(vari[in_mask], 0, 1))) if np.any(mask) else 0.0
        exgr_dominance = float(np.mean(np.clip(exgr[in_mask], 0, 255))) if np.any(mask) else 0.0
        vegetation = min(100.0, green_dominance / 1.5)
        vari_score = min(100.0, max(0.0, vari_dominance * 100))
        exgr_score = min(100.0, exgr_dominance / 1.5)
        health = min(100.0, 0.55 * vegetation + 0.45 * coverage)
        return VegetationMetrics(
            mask,
            round(coverage, 2),
            round(vegetation, 2),
            round(health, 2),
            round(vari_score, 2),
            round(exgr_score, 2),
        )

    @staticmethod
    def _otsu_mask(index: np.ndarray) -> np.ndarray:
        normalized = cv2.normalize(index, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
