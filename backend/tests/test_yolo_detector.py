from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sahi.prediction import ObjectPrediction

from app.services.yolo_detector import CONF_THRESHOLD, TILE_SIZE, YOLODetector


@pytest.fixture
def detector():
    return YOLODetector()


def blank_image(size: int) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


class TestDetectDispatch:
    # Which code path runs is decided by image size alone -- these patch
    # both branches and assert exactly one fires, without loading a model.
    def test_image_at_or_below_tile_size_uses_single_pass(self, detector):
        with patch.object(detector, "_detect_single", return_value=[]) as single, \
             patch.object(detector, "_detect_tiled", return_value=[]) as tiled:
            detector.detect(blank_image(TILE_SIZE))
        single.assert_called_once()
        tiled.assert_not_called()

    def test_image_above_tile_size_uses_tiled_pass(self, detector):
        with patch.object(detector, "_detect_single", return_value=[]) as single, \
             patch.object(detector, "_detect_tiled", return_value=[]) as tiled:
            detector.detect(blank_image(TILE_SIZE + 1))
        tiled.assert_called_once()
        single.assert_not_called()


class TestDetectTiled:
    def test_converts_sahi_predictions_to_detections(self, detector):
        fake_predictions = [
            ObjectPrediction(bbox=[10.0, 20.0, 110.0, 220.0], category_id=0, category_name="crop", score=0.81),
            ObjectPrediction(bbox=[5.5, 6.5, 45.5, 66.5], category_id=1, category_name="weed", score=0.37),
        ]
        with patch.object(YOLODetector, "_load_sahi", return_value=object()), \
             patch("sahi.predict.get_sliced_prediction", return_value=SimpleNamespace(object_prediction_list=fake_predictions)):
            detections = detector._detect_tiled(blank_image(TILE_SIZE * 2), CONF_THRESHOLD)

        assert len(detections) == 2
        first, second = detections
        assert (first.x1, first.y1, first.x2, first.y2) == (10.0, 20.0, 110.0, 220.0)
        assert first.confidence == pytest.approx(0.81)
        assert first.label == "crop"
        assert second.label == "weed"
        assert second.confidence == pytest.approx(0.37)

    def test_no_detections_returns_empty_list(self, detector):
        with patch.object(YOLODetector, "_load_sahi", return_value=object()), \
             patch("sahi.predict.get_sliced_prediction", return_value=SimpleNamespace(object_prediction_list=[])):
            assert detector._detect_tiled(blank_image(TILE_SIZE * 2), CONF_THRESHOLD) == []

    def test_passes_conf_threshold_to_sliced_prediction(self, detector):
        with patch.object(YOLODetector, "_load_sahi", return_value=object()), \
             patch("sahi.predict.get_sliced_prediction", return_value=SimpleNamespace(object_prediction_list=[])) as mock_sliced:
            detector._detect_tiled(blank_image(TILE_SIZE * 2), 0.42)
        assert mock_sliced.call_args.kwargs["confidence_threshold"] == 0.42


class TestDetectSingle:
    def test_passes_conf_threshold_to_predict(self, detector):
        mock_result = MagicMock(boxes=[])
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        with patch.object(detector, "_load", return_value=mock_model):
            detector._detect_single(blank_image(100), 0.42)
        assert mock_model.predict.call_args.kwargs["conf"] == 0.42
