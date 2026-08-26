from dataclasses import dataclass

import cv2
import numpy as np

# 2 is OpenCV's own hard floor (cv2.Stitcher can't do anything with fewer),
# 12 is this project's own ceiling -- not a Stitcher limit, a practical one:
# each added photo roughly multiplies the pairwise feature-matching cost,
# and this runs synchronously in one HTTP request on CPU with no job queue
# (see worker.py's leasing pattern, which this deliberately doesn't use --
# a same-request stitch of a handful of photos doesn't need it).
MIN_IMAGES = 2
MAX_IMAGES = 12


@dataclass(frozen=True)
class MosaicResult:
    success: bool
    mosaic: np.ndarray | None  # BGR, arbitrary size -- whatever the stitched composite came out to
    images_used: int  # how many of the input photos actually made it into the mosaic
    images_submitted: int
    warning: str | None


class MosaicStitcher:
    """Stitches multiple overlapping field photos into one wider composite
    via OpenCV's Stitcher pipeline (feature detection/matching, homography
    estimation, bundle adjustment, exposure compensation, seam blending --
    the real multi-image algorithm, not a naive pairwise-homography chain
    that would drift and double-blend overlaps).

    Deliberately NOT a georeferenced orthomosaic: there's no GPS/altitude
    data tying any pixel here to a real-world coordinate (see
    field_area_estimator.py for why that data is often unavailable even
    for a single photo), and no ground-elevation model to ortho-rectify
    against even if there were. This produces a real, algorithmically
    -stitched image in the *first* photo's own relative pixel space --
    useful for seeing more of a field at once, not for surveying it.
    """

    def __init__(self) -> None:
        self._mode = cv2.Stitcher_SCANS

    def stitch(self, images: list[np.ndarray]) -> MosaicResult:
        if len(images) < MIN_IMAGES:
            return MosaicResult(
                success=False, mosaic=None, images_used=0, images_submitted=len(images),
                warning=f"Need at least {MIN_IMAGES} photos to stitch a mosaic, got {len(images)}.",
            )
        if len(images) > MAX_IMAGES:
            return MosaicResult(
                success=False, mosaic=None, images_used=0, images_submitted=len(images),
                warning=f"Too many photos for one mosaic (max {MAX_IMAGES}, got {len(images)}) -- split into smaller batches.",
            )

        stitcher = cv2.Stitcher_create(self._mode)
        status, mosaic = stitcher.stitch(images)

        if status != cv2.Stitcher_OK:
            return MosaicResult(
                success=False, mosaic=None, images_used=0, images_submitted=len(images),
                warning=_status_message(status, len(images)),
            )

        return MosaicResult(
            success=True, mosaic=mosaic, images_used=len(images), images_submitted=len(images),
            warning=None,
        )


def _status_message(status: int, submitted: int) -> str:
    """OpenCV's Stitcher reports one of three failure codes -- translated
    here into the same honest, specific-reason style FlightComparator uses
    (see _failed there) rather than a bare status number the caller has to
    look up."""
    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        return (
            f"Not enough of these {submitted} photos share enough visual overlap to align -- "
            "OpenCV's stitcher needs each photo to clearly overlap at least one other in the set."
        )
    if status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        return "Could not find a reliable alignment between these photos -- they may not overlap enough or may not show the same field."
    if status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        return "Found overlapping features but couldn't reconcile a consistent camera arrangement across all the photos."
    return f"Stitching failed (OpenCV status code {status})."
