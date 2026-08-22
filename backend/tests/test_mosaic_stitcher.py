import cv2
import numpy as np
import pytest

from app.services.mosaic_stitcher import MAX_IMAGES, MIN_IMAGES, MosaicStitcher


def _textured_panorama(width=900, height=400, seed=0) -> np.ndarray:
    """A wide synthetic "field" with enough distinct visual texture (soil
    noise + scattered colored blobs, no repeating pattern) for OpenCV's
    Stitcher to find real, unambiguous feature matches -- a flat or
    repetitive image would give it nothing reliable to align on."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), (90, 110, 140), dtype=np.uint8)
    noise = rng.integers(-15, 15, size=image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(250):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(image, (x, y), int(rng.integers(4, 12)), color, -1)
    return image


def _overlapping_tiles(panorama: np.ndarray, n: int, overlap_fraction: float = 0.4) -> list[np.ndarray]:
    """Slices n overlapping tiles left-to-right across a wide source image
    -- real ground truth for "these photos show the same field with real
    overlap", unlike unrelated random images."""
    height, width = panorama.shape[:2]
    tile_width = int(width / (1 + (n - 1) * (1 - overlap_fraction)))
    step = int(tile_width * (1 - overlap_fraction))
    tiles = []
    for i in range(n):
        x0 = min(i * step, width - tile_width)
        tiles.append(panorama[:, x0 : x0 + tile_width].copy())
    return tiles


@pytest.fixture
def stitcher():
    return MosaicStitcher()


class TestMosaicStitcher:
    def test_stitches_overlapping_tiles_into_a_wider_composite(self, stitcher):
        panorama = _textured_panorama()
        tiles = _overlapping_tiles(panorama, n=3)

        result = stitcher.stitch(tiles)

        assert result.success
        assert result.mosaic is not None
        assert result.images_used == 3
        assert result.images_submitted == 3
        assert result.warning is None
        # A real stitch of 3 overlapping tiles should be noticeably wider
        # than any single input tile, not just a copy of one of them.
        assert result.mosaic.shape[1] > tiles[0].shape[1] * 1.5

    def test_too_few_images_fails_honestly(self, stitcher):
        result = stitcher.stitch([_textured_panorama()])

        assert result.success is False
        assert result.mosaic is None
        assert result.warning is not None
        assert str(MIN_IMAGES) in result.warning

    def test_too_many_images_fails_honestly(self, stitcher):
        images = [_textured_panorama(width=100, height=100, seed=i) for i in range(MAX_IMAGES + 1)]

        result = stitcher.stitch(images)

        assert result.success is False
        assert result.mosaic is None
        assert str(MAX_IMAGES) in result.warning

    def test_unrelated_non_overlapping_images_fail_rather_than_fabricate_a_mosaic(self, stitcher):
        rng = np.random.default_rng(1)
        image_a = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        image_b = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)

        result = stitcher.stitch([image_a, image_b])

        assert result.success is False
        assert result.mosaic is None
        assert result.warning is not None

    def test_at_min_images_boundary_still_attempts_a_stitch(self, stitcher):
        panorama = _textured_panorama()
        tiles = _overlapping_tiles(panorama, n=MIN_IMAGES)

        result = stitcher.stitch(tiles)

        assert result.success
        assert result.images_used == MIN_IMAGES
