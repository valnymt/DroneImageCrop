from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np

from app.services.pipeline import CropAnalysisPipeline, _segmentation_overlay


class TestSegmentationOverlay:
    def test_tints_only_the_masked_region(self):
        image = np.full((40, 40, 3), (90, 140, 90), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[:20, :] = 255  # top half masked, bottom half not

        overlay = _segmentation_overlay(image, mask)

        assert not np.array_equal(overlay[:20], image[:20])
        assert np.array_equal(overlay[20:], image[20:])

    def test_empty_mask_leaves_image_unchanged(self):
        image = np.full((20, 20, 3), (90, 140, 90), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)

        overlay = _segmentation_overlay(image, mask)

        assert np.array_equal(overlay, image)


def _mocked_pipeline() -> CropAnalysisPipeline:
    # Every sub-service mocked out -- these tests are about analyze()'s own
    # orchestration (does it call the right things with the right settings),
    # not the sub-services' real behavior (covered by their own test files).
    pipeline = CropAnalysisPipeline()
    pipeline.cv = MagicMock()
    pipeline.cv.load_and_preprocess.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
    pipeline.cv.vegetation_metrics.return_value = SimpleNamespace(
        green_mask=np.zeros((50, 50), dtype=np.uint8),
        coverage_percent=10.0, vegetation_score=20.0, health_score=30.0,
        heatmap=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    pipeline.detector = MagicMock()
    pipeline.detector.detect.return_value = []
    pipeline.segmenter = MagicMock()
    pipeline.segmenter.refine.return_value = np.zeros((50, 50), dtype=np.uint8)
    return pipeline


class TestPipelineSettings:
    def test_refine_segmentation_false_skips_sam(self, tmp_path):
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        pipeline.analyze(path, "Wheat", 2.0, 0.02, refine_segmentation=False)

        pipeline.segmenter.refine.assert_not_called()

    def test_refine_segmentation_true_calls_sam(self, tmp_path):
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        pipeline.analyze(path, "Wheat", 2.0, 0.02, refine_segmentation=True)

        pipeline.segmenter.refine.assert_called_once()

    def test_passes_enhance_and_conf_threshold_through(self, tmp_path):
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        pipeline.analyze(path, "Wheat", 2.0, 0.02, enhance=False, conf_threshold=0.4)

        _, preprocess_kwargs = pipeline.cv.load_and_preprocess.call_args
        assert preprocess_kwargs["enhance"] is False
        _, detect_kwargs = pipeline.detector.detect.call_args
        assert detect_kwargs["conf_threshold"] == 0.4


class TestTextureAffectsHealthScore:
    def test_patchy_texture_scores_lower_than_uniform_texture_at_same_color_health(self, tmp_path):
        # Same vegetation/coverage color signal in both cases -- only the
        # actual pixel texture differs (uniform green vs. real GLCM-patchy
        # noise) -- confirms texture is a real, non-token input to
        # health_score, not just decoration on the response.
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((80, 80, 3), dtype=np.uint8))

        uniform_pipeline = _mocked_pipeline()
        uniform_pipeline.cv.load_and_preprocess.return_value = np.full((80, 80, 3), (60, 150, 60), dtype=np.uint8)
        uniform_pipeline.segmenter.refine.return_value = np.full((80, 80), 255, dtype=np.uint8)
        uniform_result = uniform_pipeline.analyze(path, "Wheat", 2.0)

        rng = np.random.default_rng(0)
        noisy = np.full((80, 80, 3), (60, 150, 60), dtype=np.uint8)
        noise = rng.integers(-80, 80, size=noisy.shape).astype(np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        patchy_pipeline = _mocked_pipeline()
        patchy_pipeline.cv.load_and_preprocess.return_value = noisy
        patchy_pipeline.segmenter.refine.return_value = np.full((80, 80), 255, dtype=np.uint8)
        patchy_result = patchy_pipeline.analyze(path, "Wheat", 2.0)

        assert uniform_result.texture_pattern == "uniform"
        assert patchy_result.texture_pattern == "patchy"
        assert patchy_result.health_score < uniform_result.health_score

    def test_texture_fields_present_on_result(self, tmp_path):
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        assert result.texture_pattern == "mixed"  # empty mask -> neutral default
        assert result.texture_uniformity_score == 50.0


class TestTiltCorrectionWiring:
    def test_correct_tilt_false_skips_tilt_correction_entirely(self, tmp_path):
        pipeline = _mocked_pipeline()
        pipeline.tilt = MagicMock()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0, correct_tilt=False)

        pipeline.tilt.correct.assert_not_called()
        assert result.tilt_corrected is False
        assert "disabled" in result.tilt_correction_note.lower()

    def test_correct_tilt_true_by_default_calls_tilt_corrector(self, tmp_path):
        pipeline = _mocked_pipeline()
        pipeline.tilt = MagicMock()
        pipeline.tilt.correct.return_value = SimpleNamespace(
            corrected=False, image=np.zeros((50, 50, 3), dtype=np.uint8), note="Not enough visible line structure.",
        )
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        pipeline.tilt.correct.assert_called_once()
        assert result.tilt_corrected is False
        assert result.tilt_correction_note == "Not enough visible line structure."

    def test_downstream_analysis_uses_the_corrected_image_when_tilt_was_corrected(self, tmp_path):
        # The whole point of running this before everything else: once a
        # correction is applied, vegetation/detection must see the
        # corrected geometry, not the original tilted photo.
        pipeline = _mocked_pipeline()
        corrected_image = np.full((64, 96, 3), (60, 150, 60), dtype=np.uint8)
        pipeline.segmenter.refine.return_value = np.zeros((64, 96), dtype=np.uint8)
        pipeline.tilt = MagicMock()
        pipeline.tilt.correct.return_value = SimpleNamespace(corrected=True, image=corrected_image, note="Perspective corrected.")
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        veg_call_image = pipeline.cv.vegetation_metrics.call_args[0][0]
        assert veg_call_image.shape == corrected_image.shape
        assert np.array_equal(veg_call_image, corrected_image)
        assert result.tilt_corrected is True
        assert result.image_width == 96 and result.image_height == 64
        assert result.tilt_correction_note == "Perspective corrected."
