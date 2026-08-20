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
