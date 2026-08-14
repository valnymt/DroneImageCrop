from pathlib import Path

from PIL import ExifTags, Image

from app.services.field_area_estimator import SENSOR_WIDTH_35MM_MM, estimate_area_hectares

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

    def test_altitude_without_focal_length_returns_unavailable(self, tmp_path):
        path = tmp_path / "no_focal.jpg"
        _save_with_gps(path, altitude=100.0, focal_length_35mm=None)

        assert estimate_area_hectares(path) == (None, "unavailable")

    def test_focal_length_without_altitude_returns_unavailable(self, tmp_path):
        path = tmp_path / "no_altitude.jpg"
        _save_with_gps(path, altitude=None, focal_length_35mm=24)

        assert estimate_area_hectares(path) == (None, "unavailable")

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
