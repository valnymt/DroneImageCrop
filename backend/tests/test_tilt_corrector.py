import cv2
import numpy as np
import pytest

from app.services.tilt_corrector import (
    MIN_ANGLE_SEPARATION_DEG,
    TiltCorrector,
    _dominant_angle_clusters,
    _line_angle_deg,
)

SOIL_BGR = (90, 110, 140)
ROW_BGR = (60, 170, 70)


@pytest.fixture
def corrector():
    return TiltCorrector()


def _grid_field(width=700, height=500, spacing=35, seed=0) -> np.ndarray:
    """A synthetic nadir (straight-down) field photo: vertical crop rows
    and horizontal cross-furrows on noisy soil -- two genuinely orthogonal,
    genuinely parallel line families, the exact structure this module
    looks for."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), SOIL_BGR, dtype=np.uint8)
    noise = rng.integers(-15, 15, size=image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for x in range(0, width, spacing):
        cv2.line(image, (x, 0), (x, height), ROW_BGR, 6)
    for y in range(0, height, spacing * 2):
        cv2.line(image, (0, y), (width, y), (55, 150, 65), 4)
    return image


def _tilt(image: np.ndarray, top_left_shift=80, top_right_shift=-30) -> np.ndarray:
    """Simulates camera tilt by warping the top edge inward -- a trapezoid
    (keystone) distortion, exactly what a non-nadir shot of a flat field
    produces."""
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[top_left_shift, 0], [w + top_right_shift, 0], [w, h], [0, h]])
    homography = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, homography, (w, h), borderValue=SOIL_BGR)


def _row_family_angle_std(corrector: TiltCorrector, image: np.ndarray) -> float:
    """Re-detects lines on the given image and returns the angular spread
    (degrees) of its dominant line family -- the direct measure of "how
    parallel are these lines actually" that a working rectification should
    shrink."""
    mask = corrector.cv.vegetation_metrics(image).green_mask
    lines = corrector._detect_row_candidate_lines(image, mask)
    angles = np.array([_line_angle_deg(*line) for line in lines])
    peak, _ = _dominant_angle_clusters(angles)

    def circular_diff(angle: float, center: int) -> float:
        d = abs(angle - center) % 180
        return min(d, 180 - d)

    family = angles[[circular_diff(a, peak) <= MIN_ANGLE_SEPARATION_DEG / 2 for a in angles]]
    return float(family.std())


class TestTiltCorrector:
    def test_already_nadir_field_is_not_corrected(self, corrector):
        result = corrector.correct(_grid_field())

        assert result.corrected is False
        assert "nadir" in result.note.lower()

    def test_tilted_field_is_corrected(self, corrector):
        tilted = _tilt(_grid_field())

        result = corrector.correct(tilted)

        assert result.corrected is True
        assert result.note
        assert result.row_family_lines > 0 and result.cross_family_lines > 0

    def test_correction_measurably_improves_row_parallelism(self, corrector):
        # The actual proof this does real rectification, not decoration:
        # re-detecting lines on the corrected output should show the
        # dominant row family noticeably more parallel than it was in the
        # tilted input.
        tilted = _tilt(_grid_field())
        result = corrector.correct(tilted)
        assert result.corrected

        std_before = _row_family_angle_std(corrector, tilted)
        std_after = _row_family_angle_std(corrector, result.image)

        assert std_after < std_before * 0.5

    def test_more_severe_tilt_still_corrects(self, corrector):
        severely_tilted = _tilt(_grid_field(), top_left_shift=160, top_right_shift=-70)

        result = corrector.correct(severely_tilted)

        assert result.corrected is True

    def test_photo_with_no_vegetation_is_not_corrected(self, corrector):
        blank_soil = np.full((300, 400, 3), SOIL_BGR, dtype=np.uint8)

        result = corrector.correct(blank_soil)

        assert result.corrected is False
        assert result.note
        assert np.array_equal(result.image, blank_soil)

    def test_random_noise_image_is_not_corrected(self, corrector):
        rng = np.random.default_rng(3)
        noise_image = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)

        result = corrector.correct(noise_image)

        assert result.corrected is False

    def test_uncorrected_result_returns_the_original_image_unchanged(self, corrector):
        blank_soil = np.full((300, 400, 3), SOIL_BGR, dtype=np.uint8)

        result = corrector.correct(blank_soil)

        assert result.image is blank_soil or np.array_equal(result.image, blank_soil)

    def test_single_row_direction_without_a_second_family_is_not_corrected(self, corrector):
        # Only vertical rows, no cross-furrows at all -- one vanishing
        # point isn't enough for a principled rectification (see the
        # module docstring: needs two directions' vanishing line).
        rng = np.random.default_rng(7)
        image = np.full((400, 500, 3), SOIL_BGR, dtype=np.uint8)
        noise = rng.integers(-15, 15, size=image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for x in range(0, 500, 30):
            cv2.line(image, (x, 0), (x, 400), ROW_BGR, 6)

        result = corrector.correct(image)

        assert result.corrected is False
