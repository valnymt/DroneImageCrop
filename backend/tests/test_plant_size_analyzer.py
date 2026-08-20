import cv2
import numpy as np
import pytest

from app.services.plant_size_analyzer import PlantSizeAnalyzer


@pytest.fixture
def analyzer():
    return PlantSizeAnalyzer()


def _circle_mask(shape=(200, 200), center=(100, 100), radius=20) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def _rect_mask(shape=(200, 200), top_left=(20, 90), bottom_right=(180, 110)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    return mask


class TestPlantSizeAnalyzer:
    def test_no_masks_returns_none(self, analyzer):
        assert analyzer.analyze([], cm2_per_pixel=1.0) is None

    def test_masks_with_no_contours_are_skipped_not_fabricated(self, analyzer):
        blank = np.zeros((100, 100), dtype=np.uint8)

        assert analyzer.analyze([blank, blank], cm2_per_pixel=1.0) is None

    def test_identical_circles_report_perfect_uniformity(self, analyzer):
        masks = [
            _circle_mask(center=(50, 50), radius=20),
            _circle_mask(center=(150, 50), radius=20),
            _circle_mask(center=(100, 150), radius=20),
        ]

        result = analyzer.analyze(masks, cm2_per_pixel=1.0)

        assert result.plant_count == 3
        assert result.size_uniformity_score == 100.0
        assert result.min_area_cm2 == result.max_area_cm2 == result.mean_area_cm2

    def test_widely_varied_sizes_report_low_uniformity(self, analyzer):
        masks = [
            _circle_mask(center=(50, 50), radius=8),
            _circle_mask(center=(150, 50), radius=25),
            _circle_mask(center=(100, 150), radius=45),
        ]

        result = analyzer.analyze(masks, cm2_per_pixel=1.0)

        assert result.size_uniformity_score < 50
        assert result.min_area_cm2 < result.mean_area_cm2 < result.max_area_cm2

    def test_area_scales_with_cm2_per_pixel(self, analyzer):
        mask = _circle_mask(radius=10)

        result_1x = analyzer.analyze([mask], cm2_per_pixel=1.0)
        result_4x = analyzer.analyze([mask], cm2_per_pixel=4.0)

        assert result_4x.mean_area_cm2 == pytest.approx(result_1x.mean_area_cm2 * 4, rel=1e-6)

    def test_circle_has_aspect_ratio_near_one(self, analyzer):
        result = analyzer.analyze([_circle_mask(radius=30)], cm2_per_pixel=1.0)

        assert result.mean_aspect_ratio == pytest.approx(1.0, abs=0.05)

    def test_elongated_shape_has_higher_aspect_ratio(self, analyzer):
        result = analyzer.analyze([_rect_mask()], cm2_per_pixel=1.0)

        assert result.mean_aspect_ratio > 5.0

    def test_plant_count_reflects_only_masks_with_real_contours(self, analyzer):
        blank = np.zeros((100, 100), dtype=np.uint8)
        real = _circle_mask((100, 100), (50, 50), 15)

        result = analyzer.analyze([real, blank, real], cm2_per_pixel=1.0)

        assert result.plant_count == 2

    def test_single_plant_is_perfectly_uniform_by_definition(self, analyzer):
        result = analyzer.analyze([_circle_mask(radius=12)], cm2_per_pixel=2.5)

        assert result.plant_count == 1
        assert result.size_uniformity_score == 100.0
        assert result.mean_area_cm2 == result.median_area_cm2 == result.min_area_cm2 == result.max_area_cm2
