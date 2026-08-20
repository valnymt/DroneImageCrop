from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.sam_segmenter import SAMSegmenter
from app.services.schemas import Detection


def _detection(x1=0, y1=0, x2=10, y2=10) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, label="crop")


@pytest.fixture
def segmenter(tmp_path):
    # A nonexistent checkpoint path -- _load() returns None without ever
    # touching torch/mobile_sam, so these tests exercise this module's own
    # logic (not MobileSAM's) without needing the real model installed.
    return SAMSegmenter(checkpoint=tmp_path / "does_not_exist.pt")


class TestUnionMasks:
    def test_unions_multiple_boolean_masks(self):
        mask_a = np.zeros((10, 10), dtype=bool)
        mask_a[0:3, 0:3] = True
        mask_b = np.zeros((10, 10), dtype=bool)
        mask_b[7:10, 7:10] = True

        union = SAMSegmenter.union_masks([mask_a, mask_b], (10, 10))

        assert union[1, 1] == 255
        assert union[8, 8] == 255
        assert union[5, 5] == 0

    def test_empty_instance_list_produces_an_all_zero_mask(self):
        union = SAMSegmenter.union_masks([], (10, 10))

        assert union.shape == (10, 10)
        assert union.sum() == 0


class TestSegmentInstancesAndRefineWithoutACheckpoint:
    def test_segment_instances_returns_none_when_checkpoint_missing(self, segmenter):
        result = segmenter.segment_instances(np.zeros((20, 20, 3), dtype=np.uint8), [_detection()])

        assert result is None

    def test_segment_instances_returns_none_with_no_detections(self, segmenter):
        assert segmenter.segment_instances(np.zeros((20, 20, 3), dtype=np.uint8), []) is None
        assert segmenter.segment_instances(np.zeros((20, 20, 3), dtype=np.uint8), None) is None

    def test_refine_falls_back_to_initial_mask_when_checkpoint_missing(self, segmenter):
        initial_mask = np.full((20, 20), 128, dtype=np.uint8)

        result = segmenter.refine(np.zeros((20, 20, 3), dtype=np.uint8), initial_mask, [_detection()])

        assert np.array_equal(result, initial_mask)


class TestRefineIsConsistentWithSegmentInstances:
    def test_refine_equals_the_union_of_segment_instances(self, segmenter):
        # refine() must not diverge from what segment_instances()
        # produces -- pipeline.py relies on both being backed by exactly
        # the same per-plant masks, not two different code paths.
        mask_a = np.zeros((15, 15), dtype=bool)
        mask_a[2:5, 2:5] = True
        mask_b = np.zeros((15, 15), dtype=bool)
        mask_b[10:13, 10:13] = True
        fake_predictor = MagicMock()
        fake_predictor.predict.side_effect = [
            (np.array([mask_a]), None, None),
            (np.array([mask_b]), None, None),
        ] * 2  # segment_instances() and refine() each drive their own predict() calls below
        segmenter._predictor = fake_predictor

        image = np.zeros((15, 15, 3), dtype=np.uint8)
        detections = [_detection(), _detection(x1=5, y1=5, x2=15, y2=15)]

        instances = segmenter.segment_instances(image, detections)
        refined = segmenter.refine(image, np.zeros((15, 15), dtype=np.uint8), detections)

        assert np.array_equal(refined, SAMSegmenter.union_masks(instances, (15, 15)))
        assert refined[3, 3] == 255
        assert refined[11, 11] == 255
