from pathlib import Path

import numpy as np
import pytest
from PIL import ExifTags, Image

from app.services.field_area_estimator import (
    AREA_CONFIDENCE,
    DEFAULT_FOCAL_LENGTH_35MM_MM,
    ROW_SPACING_METERS,
    SENSOR_WIDTH_35MM_MM,
    area_confidence,
    estimate_area_from_row_spacing,
    estimate_area_hectares,
)

WIDTH_PX, HEIGHT_PX = 4000, 3000


def _expected_area_hectares(altitude: float, focal_length_35mm: float, width_px=WIDTH_PX, height_px=HEIGHT_PX) -> float:
    # Independent restatement of the documented GSD formula, so this test
    # verifies against the spec rather than against the implementation's
    # own arithmetic.
    gsd_cm_per_px = (SENSOR_WIDTH_35MM_MM * altitude * 100) / (focal_length_35mm * width_px)
    area_m2 = (gsd_cm_per_px / 100 * width_px) * (gsd_cm_per_px / 100 * height_px)
    return round(area_m2 / 10000, 3)


def _blank_image() -> Image.Image:
    return Image.new("RGB", (WIDTH_PX, HEIGHT_PX), color=(60, 120, 60))


def _save_with_gps(path: Path, altitude: float | None, focal_length_35mm: float | None) -> None:
    with _blank_image() as image:
        exif = image.getexif()
        if focal_length_35mm is not None:
            exif[ExifTags.Base.FocalLengthIn35mmFilm] = focal_length_35mm
        if altitude is not None:
            exif[ExifTags.IFD.GPSInfo] = {6: altitude}  # 6 = GPSAltitude
        image.save(path, format="JPEG", exif=exif)


def _xmp_packet(altitude: float) -> bytes:
    # Minimal DJI-style XMP packet -- only the RelativeAltitude attribute
    # matters, the surrounding structure just has to be well-formed enough
    # for the regex in _read_xmp_altitude to find it.
    return (
        '<?xpacket begin="?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description drone-dji:RelativeAltitude="+{altitude}"/>'
        "</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    ).encode("utf-8")


def _save_with_xmp(path: Path, altitude: float, focal_length_35mm: float) -> None:
    with _blank_image() as image:
        exif = image.getexif()
        exif[ExifTags.Base.FocalLengthIn35mmFilm] = focal_length_35mm
        image.save(path, format="JPEG", exif=exif, xmp=_xmp_packet(altitude))


def _striped_field_array(period_px: int = 40, width_px: int = 480, height_px: int = 360):
    """A synthetic aerial-looking field: vertical green crop rows on brown
    soil, spaced exactly `period_px` apart -- gives the row-spacing
    heuristic an unambiguous, known-ground-truth pattern to detect."""
    image = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    image[:, :] = (90, 110, 140)  # bare soil, BGR
    for x in range(0, width_px, period_px):
        row_width = max(2, period_px // 3)
        image[:, x : x + row_width] = (60, 170, 70)  # crop row, BGR (green-dominant)
    return image, period_px, width_px, height_px


def _striped_field_image() -> Image.Image:
    image_bgr, *_ = _striped_field_array()
    return Image.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB for PIL


class TestEstimateAreaHectares:
    def test_matches_documented_gsd_formula_via_exif_gps_altitude(self, tmp_path):
        path = tmp_path / "drone.jpg"
        _save_with_gps(path, altitude=120.5, focal_length_35mm=24)

        area, source = estimate_area_hectares(path)

        assert area == _expected_area_hectares(120.5, 24)
        assert source == "exif_gps_altitude"

    def test_falls_back_to_xmp_relative_altitude_without_gps_ifd(self, tmp_path):
        path = tmp_path / "dji.jpg"
        _save_with_xmp(path, altitude=80.0, focal_length_35mm=28)

        area, source = estimate_area_hectares(path)

        assert area == _expected_area_hectares(80.0, 28)
        assert source == "xmp_relative_altitude"

    def test_prefers_exif_gps_altitude_when_xmp_also_present(self, tmp_path):
        path = tmp_path / "both.jpg"
        with _blank_image() as image:
            exif = image.getexif()
            exif[ExifTags.Base.FocalLengthIn35mmFilm] = 24
            exif[ExifTags.IFD.GPSInfo] = {6: 100.0}
            image.save(path, format="JPEG", exif=exif, xmp=_xmp_packet(999.0))

        area, source = estimate_area_hectares(path)

        assert source == "exif_gps_altitude"
        assert area == _expected_area_hectares(100.0, 24)

    def test_no_metadata_returns_unavailable(self, tmp_path):
        path = tmp_path / "plain.jpg"
        with Image.new("RGB", (800, 600), color=(200, 200, 200)) as image:
            image.save(path, format="JPEG")

        assert estimate_area_hectares(path) == (None, "unavailable")

    def test_altitude_without_focal_length_uses_default_focal_length(self, tmp_path):
        path = tmp_path / "no_focal.jpg"
        _save_with_gps(path, altitude=100.0, focal_length_35mm=None)

        area, source = estimate_area_hectares(path)

        assert area == _expected_area_hectares(100.0, DEFAULT_FOCAL_LENGTH_35MM_MM)
        assert source == "exif_gps_altitude_default_focal"

    def test_focal_length_without_altitude_falls_through_to_row_spacing(self, tmp_path):
        # The blank test fixture has no row structure to detect, so with no
        # altitude anywhere this correctly bottoms out at "unavailable" --
        # it should NOT be confused with the row-spacing fallback itself,
        # which is tested directly below.
        path = tmp_path / "no_altitude.jpg"
        _save_with_gps(path, altitude=None, focal_length_35mm=24)

        assert estimate_area_hectares(path) == (None, "unavailable")

    def test_manual_altitude_used_when_no_metadata_present(self, tmp_path):
        path = tmp_path / "no_metadata.jpg"
        with _blank_image() as image:
            image.save(path, format="JPEG")

        area, source = estimate_area_hectares(path, manual_altitude_m=50.0)

        assert area == _expected_area_hectares(50.0, DEFAULT_FOCAL_LENGTH_35MM_MM, WIDTH_PX, HEIGHT_PX)
        assert source == "manual_altitude_default_focal"

    def test_manual_altitude_combined_with_real_focal_length(self, tmp_path):
        path = tmp_path / "focal_only.jpg"
        _save_with_gps(path, altitude=None, focal_length_35mm=24)

        area, source = estimate_area_hectares(path, manual_altitude_m=60.0)

        assert area == _expected_area_hectares(60.0, 24)
        assert source == "manual_altitude"

    def test_exif_altitude_still_takes_priority_over_manual_altitude(self, tmp_path):
        path = tmp_path / "has_gps.jpg"
        _save_with_gps(path, altitude=120.5, focal_length_35mm=24)

        area, source = estimate_area_hectares(path, manual_altitude_m=999.0)

        assert area == _expected_area_hectares(120.5, 24)
        assert source == "exif_gps_altitude"

    def test_zero_altitude_returns_unavailable(self, tmp_path):
        # GPSAltitude=0 is present-but-falsy -- must be treated as "no usable
        # reading", not fall through to a divide-by-zero in the GSD formula.
        path = tmp_path / "zero_altitude.jpg"
        _save_with_gps(path, altitude=0.0, focal_length_35mm=24)

        assert estimate_area_hectares(path) == (None, "unavailable")

    def test_corrupt_file_returns_unavailable(self, tmp_path):
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not a real jpeg")

        assert estimate_area_hectares(path) == (None, "unavailable")

    def test_missing_file_returns_unavailable(self, tmp_path):
        assert estimate_area_hectares(tmp_path / "does_not_exist.jpg") == (None, "unavailable")

    def test_falls_back_to_row_spacing_when_no_altitude_but_rows_visible(self, tmp_path):
        path = tmp_path / "rows_no_metadata.jpg"
        _striped_field_image().save(str(path), quality=95)

        area, source = estimate_area_hectares(path, crop_type="Corn")

        assert source == "row_spacing_estimate"
        assert area is not None and area > 0


class TestEstimateAreaFromRowSpacing:
    def test_detects_known_row_spacing_from_synthetic_stripes(self):
        image_bgr, period_px, width_px, height_px = _striped_field_array(period_px=40)

        area, source = estimate_area_from_row_spacing(image_bgr, "Corn")

        assert source == "row_spacing_estimate"
        expected_gsd = ROW_SPACING_METERS["Corn"] / period_px
        expected_area = round((expected_gsd * width_px) * (expected_gsd * height_px) / 10000, 3)
        # The detected lag is only ever exact to within a pixel or two, so
        # the resulting area is compared with tolerance rather than exactly.
        assert area == pytest.approx(expected_area, rel=0.15)

    def test_unknown_crop_type_uses_default_row_spacing(self):
        image_bgr, period_px, width_px, height_px = _striped_field_array(period_px=40)

        area, _ = estimate_area_from_row_spacing(image_bgr, "SomeUnknownCrop")

        expected_gsd = ROW_SPACING_METERS["_default"] / period_px
        expected_area = round((expected_gsd * width_px) * (expected_gsd * height_px) / 10000, 3)
        assert area == pytest.approx(expected_area, rel=0.15)

    def test_uniform_field_with_no_row_structure_returns_unavailable(self):
        # Solid green, no periodicity anywhere -- must not fabricate a
        # spacing out of noise.
        image_bgr = np.full((300, 400, 3), (60, 150, 60), dtype=np.uint8)

        area, source = estimate_area_from_row_spacing(image_bgr, "Corn")

        assert (area, source) == (None, "unavailable")

    def test_bare_soil_with_no_vegetation_returns_unavailable(self):
        image_bgr = np.full((300, 400, 3), (90, 110, 140), dtype=np.uint8)  # brown, BGR

        area, source = estimate_area_from_row_spacing(image_bgr, "Corn")

        assert (area, source) == (None, "unavailable")

    def test_single_vegetation_blob_does_not_false_positive_as_rows(self):
        # Regression test: a single smooth patch of vegetation (no
        # repetition at all) reliably produced ONE strong-looking FFT peak
        # in an earlier version of this heuristic -- a real photo of grass
        # was misread as "row_spacing_estimate" with an absurd 0.001 ha
        # result. A lone blob must never pass the row-detection heuristic;
        # only genuine repetition (checked in the tests above) should.
        height_px, width_px = 300, 400
        yy, xx = np.mgrid[0:height_px, 0:width_px]
        cy, cx = height_px / 2, width_px / 2
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        falloff = np.clip(1 - dist / dist.max(), 0, 1)
        image_bgr = np.zeros((height_px, width_px, 3), dtype=np.uint8)
        image_bgr[:, :, 0] = 60  # constant blue/red so only green varies with the blob
        image_bgr[:, :, 2] = 60
        image_bgr[:, :, 1] = (80 + falloff * 140).astype(np.uint8)  # green channel: soft central blob

        area, source = estimate_area_from_row_spacing(image_bgr, "Corn")

        assert (area, source) == (None, "unavailable")

    def test_smooth_illumination_gradient_does_not_false_positive_as_rows(self):
        # A left-to-right brightness gradient (vignetting, low sun angle)
        # has no row structure either -- linear detrending in
        # _dominant_periodicity_px exists specifically to reject this.
        height_px, width_px = 300, 400
        gradient = np.linspace(40, 220, width_px, dtype=np.uint8)
        image_bgr = np.zeros((height_px, width_px, 3), dtype=np.uint8)
        image_bgr[:, :, 0] = 60
        image_bgr[:, :, 2] = 60
        image_bgr[:, :, 1] = np.tile(gradient, (height_px, 1))

        area, source = estimate_area_from_row_spacing(image_bgr, "Corn")

        assert (area, source) == (None, "unavailable")


class TestAreaConfidence:
    def test_every_real_area_source_has_an_entry(self):
        # estimate_area_hectares can return any of these source strings --
        # a missing entry would silently fall through to the generic "low,
        # no range" default, hiding a real gap rather than raising one.
        real_sources = {
            "exif_gps_altitude", "exif_gps_altitude_default_focal",
            "xmp_relative_altitude", "xmp_relative_altitude_default_focal",
            "manual_altitude", "manual_altitude_default_focal",
            "row_spacing_estimate",
        }
        assert real_sources <= AREA_CONFIDENCE.keys()

    def test_gps_altitude_sources_get_no_numeric_range(self):
        # The MSL-vs-AGL gap (see the module's docstring) makes the real
        # error structurally unbounded -- any numeric range here would be
        # fake precision, not honesty.
        for source in ("exif_gps_altitude", "exif_gps_altitude_default_focal"):
            low, high, tier = area_confidence(source)
            assert (low, high) == (None, None)
            assert tier == "low"

    def test_gps_altitude_is_demoted_from_a_naive_high_confidence(self):
        # This is the actual defect this module fixes -- these sources used
        # to read identically to genuinely-measured XMP/manual altitude.
        _, _, gps_tier = area_confidence("exif_gps_altitude")
        _, _, xmp_tier = area_confidence("xmp_relative_altitude")
        assert gps_tier == "low"
        assert xmp_tier == "high"
        assert gps_tier != xmp_tier

    def test_sources_with_a_real_focal_length_get_a_tight_range(self):
        for source in ("xmp_relative_altitude", "manual_altitude"):
            low, high, tier = area_confidence(source)
            assert tier == "high"
            assert low is not None and high is not None
            assert low < 1.0 < high

    def test_default_focal_sources_get_a_wider_asymmetric_range(self):
        # Assumed lens uncertainty is genuinely asymmetric (area ~ 1/focal^2,
        # and the plausible focal range isn't centered on the default) --
        # this isn't a symmetric +/-X% guess.
        low, high, tier = area_confidence("xmp_relative_altitude_default_focal")
        assert tier == "medium"
        assert low is not None and high is not None
        assert low < 1.0 < high
        assert (high - 1.0) != pytest.approx(1.0 - low)

    def test_row_spacing_estimate_gets_a_defensible_range(self):
        low, high, tier = area_confidence("row_spacing_estimate")
        assert tier == "medium"
        assert low is not None and high is not None
        assert low < 1.0 < high

    def test_unknown_source_falls_back_to_low_confidence_no_range(self):
        low, high, tier = area_confidence("unavailable")
        assert (low, high, tier) == (None, None, "low")
        low, high, tier = area_confidence("something_new_and_unmapped")
        assert (low, high, tier) == (None, None, "low")
