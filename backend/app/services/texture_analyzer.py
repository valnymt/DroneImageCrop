from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


@dataclass(frozen=True)
class TextureMetrics:
    contrast: float
    homogeneity: float
    energy: float
    correlation: float
    # 0-100: how spatially uniform the canopy's texture is, independent of
    # its color. High = smooth/consistent texture (a healthy continuous
    # canopy, or a uniformly stressed one -- e.g. drought browns a whole
    # field evenly). Low = patchy/irregular texture, the kind disease,
    # pest damage, or lodging actually looks like up close, which a
    # color-only vegetation index can't tell apart from "just less green".
    uniformity_score: float
    pattern: str  # "uniform" | "mixed" | "patchy"


# Quantizing to this many gray levels keeps the co-occurrence matrix small
# and fast, and smooths over per-pixel sensor noise that would otherwise
# register as fake texture -- full 256-level GLCM is needlessly sensitive
# for this purpose.
GRAY_LEVELS = 32
DISTANCES = [1, 2, 4]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# Below this vegetation-pixel count, there isn't enough of a canopy region
# to say anything meaningful about its texture.
MIN_VEGETATION_PIXELS = 400


class TextureAnalyzer:
    def analyze(self, image_bgr: np.ndarray, vegetation_mask: np.ndarray) -> TextureMetrics:
        crop = self._crop_to_vegetation(image_bgr, vegetation_mask)
        if crop is None:
            return TextureMetrics(0.0, 0.0, 0.0, 0.0, 50.0, "mixed")

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        quantized = np.clip(gray.astype(np.float32) / 256 * GRAY_LEVELS, 0, GRAY_LEVELS - 1).astype(np.uint8)

        glcm = graycomatrix(quantized, distances=DISTANCES, angles=ANGLES, levels=GRAY_LEVELS, symmetric=True, normed=True)
        # Averaged across all distance/angle pairs -- a single scalar per
        # property that isn't tied to one particular scale or orientation,
        # since crop rows/canopy gaps have no reason to align with any one
        # of them.
        contrast = float(graycoprops(glcm, "contrast").mean())
        homogeneity = float(graycoprops(glcm, "homogeneity").mean())
        energy = float(graycoprops(glcm, "energy").mean())
        correlation = float(graycoprops(glcm, "correlation").mean())

        # homogeneity and energy both run 0-1 and both rise as texture gets
        # smoother/more repetitive -- averaging them gives one uniformity
        # measure without needing contrast's unbounded range renormalized.
        uniformity = max(0.0, min(100.0, 100 * (0.5 * homogeneity + 0.5 * energy)))
        pattern = "uniform" if uniformity >= 65 else "patchy" if uniformity < 35 else "mixed"

        return TextureMetrics(
            round(contrast, 3), round(homogeneity, 3), round(energy, 3), round(correlation, 3),
            round(uniformity, 2), pattern,
        )

    @staticmethod
    def _crop_to_vegetation(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Restricts texture analysis to (the bounding box around) the
        vegetation region -- soil/background texture would otherwise
        dominate the GLCM and say nothing about crop condition. A tight
        per-pixel mask isn't used because GLCM co-occurrence pairs that
        cross a masked-out boundary would register as fake high-contrast
        texture at every mask edge; the bounding box accepts a small
        amount of included soil/gap as the simpler, honest tradeoff.
        """
        if mask is None or cv2.countNonZero(mask) < MIN_VEGETATION_PIXELS:
            return None
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (y1 - y0) < 8 or (x1 - x0) < 8:
            return None
        return image_bgr[y0:y1, x0:x1]
