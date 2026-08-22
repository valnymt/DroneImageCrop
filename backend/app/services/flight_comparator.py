from dataclasses import dataclass

import cv2
import numpy as np

from app.services.opencv_processor import OpenCVProcessor
from app.services.texture_analyzer import TextureAnalyzer

# ORB over SIFT: patent-free (matters less than it used to, since SIFT's
# patent has since expired and is back in stock OpenCV, but ORB is still
# the conventional choice for this) and fast enough to run on CPU
# alongside everything else in this pipeline -- there's no accuracy need
# here that would justify SIFT's extra cost for two still-aerial photos of
# the same field, which share scale/orientation closely enough that ORB's
# rotation-invariant binary descriptors are already a comfortable match.
MAX_FEATURES = 3000
# Lowe's ratio test -- a match is only "good" if the best candidate is
# meaningfully closer than the second-best; this is what actually filters
# out ambiguous/wrong matches, not the raw distance threshold.
LOWE_RATIO = 0.75
# Below this many good matches, findHomography is unreliable regardless of
# what it returns -- a handful of coincidental matches can still produce a
# homography that "fits" them while being physically meaningless.
MIN_GOOD_MATCHES = 12
RANSAC_REPROJ_THRESHOLD = 5.0
# Fraction of "good" matches RANSAC must accept as inliers of the same
# homography for the alignment to be trusted -- a low inlier ratio means
# the good matches don't actually agree on a single transform (e.g. two
# unrelated photos, or a photo with a repetitive pattern confusing the
# matcher), even though MIN_GOOD_MATCHES was cleared.
MIN_INLIER_RATIO = 0.30
# Bounding-box side length, as a fraction of the analyzed image size,
# below which two changed-vegetation pixels are treated as noise rather
# than a real patch -- individual misaligned pixels at the seam of the
# warp are expected and shouldn't read as "detected changes".
MIN_PATCH_FRACTION = 0.0015
# Below this many points of uniformity_score change, GLCM's own run-to-run
# noise (lighting, compression, slightly different crop framing after the
# homography warp) is a more likely explanation than a real texture shift
# in the canopy -- this isn't a validated clinical threshold, just the
# point below which "the number moved" stops meaning "something changed".
MIN_MEANINGFUL_TEXTURE_SHIFT = 8.0


@dataclass(frozen=True)
class ComparisonResult:
    alignment_ok: bool
    keypoints_matched: int
    inlier_ratio: float  # 0-100
    growth_percent: float  # 0-100, of the overlapping analyzed region
    loss_percent: float  # 0-100
    unchanged_percent: float  # 0-100
    diff_overlay: np.ndarray  # BGR, "after" image tinted green(growth)/red(loss)
    # None whenever either photo's overlap-region vegetation is too sparse
    # for TextureAnalyzer to say anything (see MIN_VEGETATION_PIXELS) -- an
    # absent number, not a fabricated 50.0 "mixed" default bleeding into a
    # two-photo comparison where it would misleadingly look measured.
    texture_uniformity_before: float | None
    texture_uniformity_after: float | None
    # Plain-language read of the texture_uniformity delta, restricted to
    # the same real, already-uploaded pair of photos -- not a diagnosis
    # and not compared against any invented "healthy crop" baseline, just
    # what changed in the canopy's own texture between these two flights.
    texture_shift_note: str | None
    warning: str | None


class FlightComparator:
    def __init__(self) -> None:
        self.cv = OpenCVProcessor()
        self.texture = TextureAnalyzer()
        self._orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def compare(self, before_bgr: np.ndarray, after_bgr: np.ndarray) -> ComparisonResult:
        before_gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)

        kp_before, desc_before = self._orb.detectAndCompute(before_gray, None)
        kp_after, desc_after = self._orb.detectAndCompute(after_gray, None)
        if desc_before is None or desc_after is None or len(kp_before) < 2 or len(kp_after) < 2:
            return self._failed("Not enough distinguishable features found in one or both photos to align them.")

        # knnMatch(k=2) is what Lowe's ratio test needs -- the best match
        # and the runner-up, per query descriptor.
        raw_matches = self._matcher.knnMatch(desc_before, desc_after, k=2)
        good_matches = [m for m, n in raw_matches if len(raw_matches) and m.distance < LOWE_RATIO * n.distance]
        if len(good_matches) < MIN_GOOD_MATCHES:
            return self._failed(
                f"Only {len(good_matches)} reliable matching features found between the two photos "
                f"(need at least {MIN_GOOD_MATCHES}) -- they may not show the same field, or overlap too little."
            )

        src_pts = np.float32([kp_before[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_after[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        if homography is None:
            return self._failed("Could not compute a reliable alignment between the two photos.")

        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / len(good_matches)
        if inlier_ratio < MIN_INLIER_RATIO:
            return self._failed(
                f"The matching features didn't agree on a consistent alignment "
                f"({inlier_count}/{len(good_matches)} inliers) -- these may not be the same field."
            )

        h, w = after_bgr.shape[:2]
        # Warps "before" into "after"'s frame (not the other way around) so
        # the diff overlay can be drawn directly on the more recent photo
        # without a second warp.
        warped_before = cv2.warpPerspective(before_bgr, homography, (w, h))
        # warpPerspective leaves un-covered pixels at zero -- that's also a
        # legal BGR value a real photo could have, so the overlap region is
        # tracked separately via warping an all-white mask the same way,
        # rather than inferring it from warped_before's pixel values.
        overlap_mask = cv2.warpPerspective(np.full((*before_bgr.shape[:2],), 255, dtype=np.uint8), homography, (w, h))
        overlap_mask = cv2.erode(overlap_mask, np.ones((5, 5), np.uint8))  # pulls warp-edge artifacts back inside the overlap

        veg_before = self.cv.vegetation_metrics(warped_before).green_mask
        veg_after = self.cv.vegetation_metrics(after_bgr).green_mask
        in_overlap = overlap_mask > 0

        # Texture only within the overlap -- outside it, "before" and
        # "after" aren't even looking at the same ground, so a texture
        # delta there would compare two unrelated regions, not a real
        # before/after change.
        veg_before_overlap_mask = (((veg_before > 0) & in_overlap) * 255).astype(np.uint8)
        veg_after_overlap_mask = (((veg_after > 0) & in_overlap) * 255).astype(np.uint8)
        texture_before = self.texture.analyze(warped_before, veg_before_overlap_mask)
        texture_after = self.texture.analyze(after_bgr, veg_after_overlap_mask)
        # TextureAnalyzer itself falls back to a token "mixed"/50.0 result
        # below MIN_VEGETATION_PIXELS -- checked again here since that
        # placeholder shouldn't be reported as a measured before/after
        # texture value in a comparison.
        uniformity_before = texture_before.uniformity_score if cv2.countNonZero(veg_before_overlap_mask) >= 400 else None
        uniformity_after = texture_after.uniformity_score if cv2.countNonZero(veg_after_overlap_mask) >= 400 else None
        texture_shift_note = self._texture_shift_note(uniformity_before, uniformity_after)

        grew = in_overlap & (veg_before == 0) & (veg_after > 0)
        lost = in_overlap & (veg_before > 0) & (veg_after == 0)
        grew, lost = self._drop_small_patches(grew, MIN_PATCH_FRACTION), self._drop_small_patches(lost, MIN_PATCH_FRACTION)

        overlap_px = int(in_overlap.sum())
        if overlap_px == 0:
            return self._failed("The two photos don't overlap enough after alignment to compare.")
        growth_pct = round(100 * grew.sum() / overlap_px, 2)
        loss_pct = round(100 * lost.sum() / overlap_px, 2)

        overlay = after_bgr.copy()
        overlay[grew] = (60, 220, 60)   # green: new vegetation since "before"
        overlay[lost] = (50, 50, 220)   # red: vegetation lost since "before"
        # Areas outside the overlap are dimmed -- they were never actually
        # compared, so leaving them looking equally "analyzed" would be
        # misleading about what the diff covers.
        overlay[~in_overlap] = (overlay[~in_overlap].astype(np.float32) * 0.35).astype(np.uint8)

        return ComparisonResult(
            alignment_ok=True,
            keypoints_matched=inlier_count,
            inlier_ratio=round(100 * inlier_ratio, 1),
            growth_percent=growth_pct,
            loss_percent=loss_pct,
            unchanged_percent=round(100 - growth_pct - loss_pct, 2),
            diff_overlay=overlay,
            texture_uniformity_before=uniformity_before,
            texture_uniformity_after=uniformity_after,
            texture_shift_note=texture_shift_note,
            warning=None,
        )

    @staticmethod
    def _texture_shift_note(before: float | None, after: float | None) -> str | None:
        """Plain-language read of the canopy texture change between these
        two specific photos -- not a diagnosis (see TextureMetrics'
        uniformity_score docs: texture flags a pattern, it doesn't name a
        cause) and not compared to any external "healthy crop" baseline,
        just what actually shifted between this field's own two flights.
        """
        if before is None or after is None:
            return None
        delta = after - before
        if abs(delta) < MIN_MEANINGFUL_TEXTURE_SHIFT:
            return "Canopy texture is about the same between the two photos -- no meaningful shift detected."
        if delta < 0:
            return (
                f"Canopy texture got {abs(delta):.0f} points more patchy/irregular since the earlier photo "
                "-- worth a closer look at the newly-uneven areas (this flags a pattern, not a cause; disease, "
                "pest damage, and lodging can all look like this)."
            )
        return (
            f"Canopy texture got {delta:.0f} points more uniform since the earlier photo -- consistent with "
            "more even canopy fill or recovery, though a uniformly worsening field (e.g. drought) can also "
            "look smoother, not just a healthier one."
        )

    @staticmethod
    def _drop_small_patches(mask: np.ndarray, min_fraction: float) -> np.ndarray:
        """Removes connected components smaller than min_fraction of the
        mask's area -- isolated pixels from warp-edge misalignment aren't
        a "detected change", a real patch of several dozen contiguous
        pixels is."""
        min_area = max(4, int(min_fraction * mask.size))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(mask)
        for label in range(1, count):  # label 0 is background
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                keep |= labels == label
        return keep

    @staticmethod
    def _failed(reason: str) -> "ComparisonResult":
        return ComparisonResult(
            alignment_ok=False, keypoints_matched=0, inlier_ratio=0.0,
            growth_percent=0.0, loss_percent=0.0, unchanged_percent=0.0,
            diff_overlay=np.zeros((1, 1, 3), dtype=np.uint8),
            texture_uniformity_before=None, texture_uniformity_after=None, texture_shift_note=None,
            warning=reason,
        )
