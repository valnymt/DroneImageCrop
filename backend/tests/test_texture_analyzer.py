import numpy as np
import pytest

from app.services.texture_analyzer import MIN_VEGETATION_PIXELS, TextureAnalyzer


@pytest.fixture
def analyzer():
    return TextureAnalyzer()


def _full_mask(height=100, width=100) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _noisy_patch(seed=0, spread=80, height=100, width=100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), (60, 150, 60), dtype=np.uint8)
    noise = rng.integers(-spread, spread, size=base.shape).astype(np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class TestTextureAnalyzer:
    def test_uniform_patch_scores_high_and_labeled_uniform(self, analyzer):
        image = np.full((100, 100, 3), (60, 150, 60), dtype=np.uint8)

        result = analyzer.analyze(image, _full_mask())

        assert result.uniformity_score == 100.0
        assert result.pattern == "uniform"
        assert result.homogeneity == 1.0
        assert result.energy == 1.0

    def test_noisy_patchy_texture_scores_low_and_labeled_patchy(self, analyzer):
        image = _noisy_patch()

        result = analyzer.analyze(image, _full_mask())

        assert result.uniformity_score < 35
        assert result.pattern == "patchy"
        assert result.contrast > 0

    def test_more_speckled_noise_scores_lower_than_mild_noise(self, analyzer):
        # Directional check rather than exact values -- confirms the score
        # actually tracks texture roughness, not just "any noise at all".
        mild = analyzer.analyze(_noisy_patch(seed=1, spread=15), _full_mask())
        severe = analyzer.analyze(_noisy_patch(seed=1, spread=100), _full_mask())

        assert severe.uniformity_score < mild.uniformity_score

    def test_empty_mask_returns_neutral_mixed_default(self, analyzer):
        image = np.full((100, 100, 3), (60, 150, 60), dtype=np.uint8)
        empty_mask = np.zeros((100, 100), dtype=np.uint8)

        result = analyzer.analyze(image, empty_mask)

        assert result.uniformity_score == 50.0
        assert result.pattern == "mixed"

    def test_mask_below_minimum_vegetation_pixels_returns_neutral_default(self, analyzer):
        image = np.full((100, 100, 3), (60, 150, 60), dtype=np.uint8)
        sparse_mask = np.zeros((100, 100), dtype=np.uint8)
        sparse_mask[:5, :5] = 255  # 25 px, well under MIN_VEGETATION_PIXELS
        assert 25 < MIN_VEGETATION_PIXELS

        result = analyzer.analyze(image, sparse_mask)

        assert result.pattern == "mixed"
        assert result.uniformity_score == 50.0

    def test_analysis_is_restricted_to_the_vegetation_region(self, analyzer):
        # A noisy soil background surrounding a uniform vegetation patch
        # must not leak into the texture reading -- only the masked
        # (bounding-box-cropped) region should be analyzed.
        image = _noisy_patch(seed=2, spread=120, height=200, width=200)
        mask = np.zeros((200, 200), dtype=np.uint8)
        image[60:140, 60:140] = (60, 150, 60)  # uniform vegetation patch in the center
        mask[60:140, 60:140] = 255

        result = analyzer.analyze(image, mask)

        assert result.pattern == "uniform"

    def test_result_fields_are_json_serializable_types(self, analyzer):
        result = analyzer.analyze(np.full((100, 100, 3), (60, 150, 60), dtype=np.uint8), _full_mask())

        assert isinstance(result.uniformity_score, float)
        assert isinstance(result.pattern, str)
