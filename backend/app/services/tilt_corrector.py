from dataclasses import dataclass

import cv2
import numpy as np

from app.services.opencv_processor import OpenCVProcessor

# Detected line segments shorter than this (px) are usually noise/texture
# edges, not real row lines.
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 12
# Need real line structure at all before attempting anything -- a photo of
# a dense, gapless canopy with no visible row pattern has nothing for this
# to work with, and that's the common case, not an error.
MIN_LINES_TOTAL = 20
# A vanishing point estimated from fewer lines than this is just fitting
# noise, not a real direction.
MIN_FAMILY_LINES = 6
# The two dominant angle clusters must together account for at least this
# fraction of all detected lines -- otherwise there's too much scatter
# (shadows, dropzone edges, weeds) to trust either cluster as real rows.
MIN_COMBINED_FRACTION = 0.35
# Clusters closer together than this (in image-space angle, degrees) are
# probably the same direction split by noise across a bin boundary, not
# two genuinely different row directions.
MIN_ANGLE_SEPARATION_DEG = 15.0
# The vanishing line IS the line at infinity, (0, 0, 1), exactly when a
# photo has no projective distortion to correct -- so what actually
# indicates "already nadir" is the first two coefficients being near zero
# (they're what drives any real warp), not the third. Genuine tilt in
# these synthetic-grid tests produces values from ~1e-5 upward; this stays
# well under that so only true no-distortion cases are skipped.
NEAR_INFINITY_EPS = 1e-6
# Refuses a correction whose output canvas would need to be more than this
# many times the input's area -- a legitimate tilt correction stays in this
# range; anything larger means the vanishing-point fit was numerically
# unstable (near-degenerate line geometry), not a real extreme tilt.
MAX_OUTPUT_AREA_RATIO = 6.0


@dataclass(frozen=True)
class TiltCorrectionResult:
    corrected: bool
    image: np.ndarray
    row_family_lines: int
    cross_family_lines: int
    # Always set -- what happened and why, whether corrected or not. Same
    # "never leave the caller guessing" pattern as area_source/
    # texture_pattern elsewhere in this pipeline.
    note: str


def _line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Undirected orientation in [0, 180) -- a line and its reverse are the
    same line."""
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _line_homogeneous(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    p1, p2 = np.array([x1, y1, 1.0]), np.array([x2, y2, 1.0])
    line = np.cross(p1, p2)
    norm = np.linalg.norm(line[:2])
    return line / norm if norm > 1e-9 else line


def _vanishing_point(lines: np.ndarray) -> np.ndarray:
    """Least-squares vanishing point for a family of (approximately)
    concurrent lines: the point that minimizes total algebraic distance to
    every line in the family, via the null space of the stacked line-
    coefficient matrix -- far more robust to noisy line detections than
    intersecting pairs of lines directly."""
    coeffs = np.array([_line_homogeneous(*line) for line in lines])
    _, _, vt = np.linalg.svd(coeffs)
    return vt[-1]


def _dominant_angle_clusters(angles: np.ndarray) -> tuple[int, int] | None:
    """Finds the two strongest, sufficiently-distinct orientation clusters
    among detected line angles. Returns their (circular) peak angles in
    degrees, or None if there aren't two distinct enough clusters."""
    hist = np.zeros(180)
    for angle in angles:
        hist[int(angle) % 180] += 1
    # Circular smoothing (angles wrap at 180) so a real cluster spread
    # across adjacent bins still reads as one peak.
    smoothed = np.array([hist[(i - 2) : (i + 3)].sum() if i >= 2 and i < 178 else
                          (np.take(hist, range(i - 2, i + 3), mode="wrap")).sum() for i in range(180)])
    first_peak = int(np.argmax(smoothed))

    def circular_diff(a: int, b: int) -> float:
        d = abs(a - b) % 180
        return min(d, 180 - d)

    candidates = [i for i in range(180) if circular_diff(i, first_peak) >= MIN_ANGLE_SEPARATION_DEG]
    if not candidates:
        return None
    second_peak = max(candidates, key=lambda i: smoothed[i])
    if smoothed[second_peak] < 1:
        return None
    return first_peak, second_peak


class TiltCorrector:
    def __init__(self) -> None:
        self.cv = OpenCVProcessor()

    def correct(self, image_bgr: np.ndarray) -> TiltCorrectionResult:
        vegetation_mask = self.cv.vegetation_metrics(image_bgr).green_mask
        lines = self._detect_row_candidate_lines(image_bgr, vegetation_mask)
        if lines is None or len(lines) < MIN_LINES_TOTAL:
            return self._uncorrected(image_bgr, "Not enough visible line structure (crop rows, furrows) to determine tilt.")

        angles = np.array([_line_angle_deg(*line) for line in lines])
        clusters = _dominant_angle_clusters(angles)
        if clusters is None:
            return self._uncorrected(image_bgr, "Detected lines don't form two distinct enough directions to determine tilt.")
        angle_a, angle_b = clusters

        def circular_diff(angle: float, center: int) -> float:
            d = abs(angle - center) % 180
            return min(d, 180 - d)

        family_a = lines[[circular_diff(a, angle_a) <= MIN_ANGLE_SEPARATION_DEG / 2 for a in angles]]
        family_b = lines[[circular_diff(a, angle_b) <= MIN_ANGLE_SEPARATION_DEG / 2 for a in angles]]
        if len(family_a) < MIN_FAMILY_LINES or len(family_b) < MIN_FAMILY_LINES:
            return self._uncorrected(image_bgr, "Found two candidate row directions, but too few lines in one of them to trust.")
        if (len(family_a) + len(family_b)) / len(lines) < MIN_COMBINED_FRACTION:
            return self._uncorrected(image_bgr, "Too much scatter in the detected lines to confidently isolate row structure.")

        v1, v2 = _vanishing_point(family_a), _vanishing_point(family_b)
        vanishing_line = np.cross(v1, v2)
        norm = np.linalg.norm(vanishing_line)
        if norm < 1e-9:
            return self._uncorrected(image_bgr, "The two detected row directions were too close to distinguish a vanishing line.")
        vanishing_line = vanishing_line / norm

        if np.linalg.norm(vanishing_line[:2]) < NEAR_INFINITY_EPS:
            return self._uncorrected(image_bgr, "Photo already appears close to straight-down (nadir); no correction needed.")

        # Standard affine-rectification construction (Hartley & Zisserman):
        # maps the estimated vanishing line to the line at infinity,
        # removing the image's projective (perspective/keystone)
        # distortion. An affine ambiguity (residual shear/aspect) remains
        # -- this corrects the dominant tilt distortion, not a full metric
        # reconstruction, which would additionally need the two directions'
        # known real-world orthogonality enforced via the circular points.
        homography = np.array([[1, 0, 0], [0, 1, 0], vanishing_line])

        corrected = self._warp_full_frame(image_bgr, homography)
        if corrected is None:
            return self._uncorrected(image_bgr, "The estimated correction was too extreme to be numerically reliable.")

        note = (
            f"Perspective corrected using {len(family_a)} row-direction and {len(family_b)} "
            "cross-direction lines to estimate and remove camera tilt."
        )
        return TiltCorrectionResult(True, corrected, len(family_a), len(family_b), note)

    def _detect_row_candidate_lines(self, image_bgr: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray | None:
        # Edge detection runs on the vegetation mask itself, not raw
        # grayscale luma -- soil and canopy are frequently near-isoluminant
        # (clearly different in color, barely different in brightness),
        # which starves plain grayscale Canny of anything to find. The
        # mask boundary *is* exactly the row/furrow edge this is looking
        # for, since it's the same vegetation index already used
        # everywhere else in the pipeline.
        if vegetation_mask is None or cv2.countNonZero(vegetation_mask) == 0:
            return None
        smoothed_mask = cv2.medianBlur(vegetation_mask, 5)  # strips single-pixel mask noise before edge detection
        edges = cv2.Canny(smoothed_mask, 50, 150)
        raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
        if raw is None:
            return None
        return raw.reshape(-1, 4).astype(np.float64)

    @staticmethod
    def _warp_full_frame(image_bgr: np.ndarray, homography: np.ndarray) -> np.ndarray | None:
        h, w = image_bgr.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.all(np.isfinite(warped_corners)):
            return None
        x_min, y_min = warped_corners.min(axis=0)
        x_max, y_max = warped_corners.max(axis=0)
        out_w, out_h = x_max - x_min, y_max - y_min
        if out_w <= 0 or out_h <= 0 or (out_w * out_h) > MAX_OUTPUT_AREA_RATIO * w * h:
            return None
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        final_homography = translation @ homography
        return cv2.warpPerspective(image_bgr, final_homography, (int(round(out_w)), int(round(out_h))))

    @staticmethod
    def _uncorrected(image_bgr: np.ndarray, note: str) -> TiltCorrectionResult:
        return TiltCorrectionResult(False, image_bgr, 0, 0, note)
