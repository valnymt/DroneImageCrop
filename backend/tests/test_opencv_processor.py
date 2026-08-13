import cv2
import numpy as np
import pytest

from app.services.opencv_processor import OpenCVProcessor

GREEN_BGR = (30, 200, 30)
SOIL_BGR = (40, 90, 150)


@pytest.fixture
def processor():
    return OpenCVProcessor()


def two_tone_bgr(color_a, color_b, size=100):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, : size // 2] = color_a
    image[:, size // 2 :] = color_b
    return image


def patch_bgr(base_color, patch_color, size=100, patch_size=20):
    image = np.full((size, size, 3), base_color, dtype=np.uint8)
    image[:patch_size, :patch_size] = patch_color
    return image


class TestVegetationMetrics:
    def test_half_green_half_soil_reports_partial_coverage(self, processor):
        image = two_tone_bgr(GREEN_BGR, SOIL_BGR)
        metrics = processor.vegetation_metrics(image)
        assert 30 <= metrics.coverage_percent <= 70

    def test_mostly_soil_with_small_green_patch_scores_low_coverage(self, processor):
        image = patch_bgr(SOIL_BGR, GREEN_BGR)
        metrics = processor.vegetation_metrics(image)
        assert metrics.coverage_percent < 30

    def test_more_coverage_scores_a_higher_health(self, processor):
        # vegetation_score measures the greenness of pixels already inside
        # the mask, not how much of the image they cover -- both scenarios
        # use the same pure green, so only coverage_percent and (through
        # it) health_score should differ.
        mixed = processor.vegetation_metrics(two_tone_bgr(GREEN_BGR, SOIL_BGR))
        sparse = processor.vegetation_metrics(patch_bgr(SOIL_BGR, GREEN_BGR))
        assert mixed.coverage_percent > sparse.coverage_percent
        assert mixed.health_score > sparse.health_score

    def test_scores_stay_within_bounds(self, processor):
        image = two_tone_bgr(GREEN_BGR, SOIL_BGR)
        metrics = processor.vegetation_metrics(image)
        for value in (
            metrics.coverage_percent,
            metrics.vegetation_score,
            metrics.health_score,
            metrics.vari_score,
            metrics.exgr_score,
        ):
            assert 0 <= value <= 100

    def test_mask_matches_image_dimensions(self, processor):
        image = two_tone_bgr(GREEN_BGR, SOIL_BGR, size=64)
        metrics = processor.vegetation_metrics(image)
        assert metrics.green_mask.shape == (64, 64)
        assert metrics.green_mask.dtype == np.uint8

    def test_majority_vote_rejects_index_disagreement(self, processor, monkeypatch):
        # If only one of the three indices would call a region vegetation,
        # the 2-of-3 majority vote should reject it -- this is the whole
        # point of combining ExG/VARI/ExGR instead of using ExG alone.
        image = two_tone_bgr(GREEN_BGR, SOIL_BGR, size=64)
        calls = []
        original = OpenCVProcessor._otsu_mask

        def spy(index):
            mask = original(index)
            calls.append(mask)
            return mask

        monkeypatch.setattr(OpenCVProcessor, "_otsu_mask", staticmethod(spy))
        processor.vegetation_metrics(image)
        assert len(calls) == 3  # ExG, VARI, ExGR each thresholded independently


class TestLoadAndPreprocess:
    def test_raises_value_error_on_missing_file(self, processor, tmp_path):
        with pytest.raises(ValueError):
            processor.load_and_preprocess(tmp_path / "does-not-exist.jpg")

    def test_downscales_images_above_max_side(self, processor, tmp_path):
        large = np.full((1200, 2000, 3), GREEN_BGR, dtype=np.uint8)
        path = tmp_path / "large.jpg"
        cv2.imwrite(str(path), large)

        result = processor.load_and_preprocess(path, max_side=800)

        assert max(result.shape[:2]) <= 800

    def test_leaves_small_images_unscaled(self, processor, tmp_path):
        small = np.full((100, 150, 3), GREEN_BGR, dtype=np.uint8)
        path = tmp_path / "small.jpg"
        cv2.imwrite(str(path), small)

        result = processor.load_and_preprocess(path, max_side=800)

        assert result.shape[:2] == (100, 150)
