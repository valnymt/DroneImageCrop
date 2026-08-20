import cv2
import numpy as np
import pytest

from app.services.flight_comparator import MIN_GOOD_MATCHES, FlightComparator


@pytest.fixture
def comparator():
    return FlightComparator()


def _textured_field(width=400, height=300, seed=0) -> np.ndarray:
    """A synthetic aerial-looking field with enough texture (soil noise +
    randomly placed colored blobs) for ORB to find real keypoints -- a
    flat solid color has none, which would test "no features" rather than
    "same field, different time"."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), (90, 110, 140), dtype=np.uint8)
    noise = rng.integers(-20, 20, size=image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(50):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(image, (x, y), int(rng.integers(3, 9)), color, -1)
    image[:, : width // 2] = (60, 170, 70)  # vegetation on the left half
    return image


class TestFlightComparator:
    def test_identical_photos_report_no_growth_or_loss(self, comparator):
        field = _textured_field()

        result = comparator.compare(field, field.copy())

        assert result.alignment_ok
        assert result.growth_percent == 0.0
        assert result.loss_percent == 0.0
        assert result.warning is None

    def test_detects_new_growth_patch(self, comparator):
        before = _textured_field()
        after = before.copy()
        after[100:180, 280:360] = (60, 170, 70)  # bare soil -> vegetation

        result = comparator.compare(before, after)

        assert result.alignment_ok
        assert result.growth_percent > 0
        assert result.loss_percent < result.growth_percent

    def test_detects_vegetation_loss_patch(self, comparator):
        before = _textured_field()
        after = before.copy()
        after[50:110, 40:110] = (90, 110, 140)  # vegetation -> bare soil

        result = comparator.compare(before, after)

        assert result.alignment_ok
        assert result.loss_percent > 0
        assert result.growth_percent < result.loss_percent

    def test_aligns_through_rotation_translation_and_scale(self, comparator):
        # The realistic case: two drone passes are never pixel-identically
        # framed. Confirms the homography step actually corrects for
        # camera movement rather than just working on identical framing.
        before = _textured_field(width=500, height=400, seed=1)
        after_raw = before.copy()
        after_raw[120:200, 350:430] = (60, 170, 70)  # growth
        after_raw[60:130, 50:130] = (90, 110, 140)  # loss

        h, w = after_raw.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.05)
        matrix[0, 2] += 15
        matrix[1, 2] -= 10
        after = cv2.warpAffine(after_raw, matrix, (w, h), borderValue=(90, 110, 140))

        result = comparator.compare(before, after)

        assert result.alignment_ok
        assert result.inlier_ratio > 50
        assert result.growth_percent > 0
        assert result.loss_percent > 0

    def test_featureless_images_fail_honestly_rather_than_fabricate_a_diff(self, comparator):
        blank = np.full((200, 200, 3), (60, 150, 60), dtype=np.uint8)

        result = comparator.compare(blank, blank)

        assert result.alignment_ok is False
        assert result.warning is not None
        assert result.growth_percent == 0.0
        assert result.loss_percent == 0.0

    def test_unrelated_photos_fail_alignment_rather_than_produce_a_bogus_diff(self, comparator):
        rng = np.random.default_rng(5)
        image_a = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        image_b = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)

        result = comparator.compare(image_a, image_b)

        assert result.alignment_ok is False
        assert result.warning is not None

    def test_diff_overlay_matches_after_photo_dimensions_when_aligned(self, comparator):
        before = _textured_field()
        after = _textured_field(seed=2)  # different texture, same vegetation layout -- still same field size

        result = comparator.compare(before, after)

        if result.alignment_ok:
            assert result.diff_overlay.shape[:2] == after.shape[:2]

    def test_isolated_single_pixel_noise_does_not_count_as_a_change(self, comparator):
        # A handful of scattered single-pixel mismatches from warp-edge
        # interpolation shouldn't be reported as "detected changes" --
        # only real contiguous patches should move growth/loss off zero.
        before = _textured_field(width=300, height=300, seed=3)
        after = before.copy()
        rng = np.random.default_rng(9)
        for _ in range(5):
            x, y = int(rng.integers(0, 300)), int(rng.integers(150, 300))
            after[y, x] = (90, 110, 140)  # single bare-soil pixel inside the vegetated half

        result = comparator.compare(before, after)

        assert result.alignment_ok
        assert result.loss_percent == 0.0

    def test_min_good_matches_constant_is_a_real_floor(self):
        # Documents the module's own intent -- if this constant is ever
        # dropped to something trivially small, that's a real regression in
        # how confidently "same field" gets asserted, not a style nit.
        assert MIN_GOOD_MATCHES >= 8
