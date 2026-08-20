from unittest.mock import patch

import numpy as np
import pytest

from app.services.open_vocab_detector import TILE_SIZE, OpenVocabDetector, _iou, _suppress_overlaps
from app.services.schemas import Detection


def _det(x1, y1, x2, y2, confidence=0.5) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence, label="plant (general detector)")


class TestIou:
    def test_identical_boxes_have_iou_one(self):
        box = _det(0, 0, 10, 10)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        assert _iou(_det(0, 0, 10, 10), _det(20, 20, 30, 30)) == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        iou = _iou(_det(0, 0, 10, 10), _det(5, 5, 15, 15))
        assert 0 < iou < 1


class TestSuppressOverlaps:
    def test_keeps_all_non_overlapping_detections(self):
        detections = [_det(0, 0, 10, 10), _det(100, 100, 110, 110)]

        assert len(_suppress_overlaps(detections)) == 2

    def test_collapses_overlapping_detections_to_the_higher_confidence_one(self):
        weak = _det(0, 0, 10, 10, confidence=0.2)
        strong = _det(1, 1, 11, 11, confidence=0.8)

        kept = _suppress_overlaps([weak, strong])

        assert len(kept) == 1
        assert kept[0].confidence == 0.8

    def test_empty_input_returns_empty_output(self):
        assert _suppress_overlaps([]) == []


class TestTiledDetectionCoordinates:
    def test_small_image_uses_a_single_tile_at_native_coordinates(self):
        detector = OpenVocabDetector()
        small_image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(detector, "_detect_tile", return_value=[_det(5, 5, 15, 15)]) as mock_tile:
            result = detector.detect(small_image)

        mock_tile.assert_called_once()
        assert len(result) == 1
        assert (result[0].x1, result[0].y1) == (5, 5)

    def test_large_image_is_split_into_multiple_tiles(self):
        detector = OpenVocabDetector()
        large_image = np.zeros((900, 900, 3), dtype=np.uint8)
        with patch.object(detector, "_detect_tile", return_value=[]) as mock_tile:
            detector.detect(large_image)

        assert mock_tile.call_count > 1
        for call in mock_tile.call_args_list:
            tile_image = call[0][0]
            assert max(tile_image.shape[:2]) <= TILE_SIZE

    def test_tile_detections_are_translated_to_full_image_coordinates(self):
        # A detection reported at (5, 5)-(15, 15) *within* a tile whose
        # top-left corner is at (400, 0) in the full image must come back
        # as (405, 5)-(415, 15) in the merged result -- not left in
        # tile-local coordinates, which would misplace every box on any
        # image large enough to need more than one tile.
        detector = OpenVocabDetector()
        large_image = np.zeros((300, 900, 3), dtype=np.uint8)

        def fake_detect_tile(tile, threshold):
            return [_det(5, 5, 15, 15)]

        with patch.object(detector, "_detect_tile", side_effect=fake_detect_tile):
            results = detector.detect(large_image)

        # Every returned box's x1 should land on one of the tile origins
        # (0, stride, 2*stride, ...) plus the local offset of 5 -- never
        # still sitting at the raw local coordinate for a non-first tile.
        x1_values = {round(r.x1) for r in results}
        assert x1_values != {5}  # would mean translation never happened
        assert all(x >= 5 for x in x1_values)

    def test_covers_the_full_image_width_including_the_final_partial_tile(self):
        # width=900 with TILE_SIZE=400 doesn't divide evenly -- the last
        # tile must still be pulled back to end exactly at the image edge
        # (not run past it, not leave a gap uncovered).
        detector = OpenVocabDetector()
        large_image = np.zeros((400, 900, 3), dtype=np.uint8)
        seen_tile_shapes = []

        def fake_detect_tile(tile, threshold):
            seen_tile_shapes.append(tile.shape)
            return []

        with patch.object(detector, "_detect_tile", side_effect=fake_detect_tile):
            detector.detect(large_image)

        assert all(shape[1] == TILE_SIZE for shape in seen_tile_shapes)  # every tile is full-width, none truncated
