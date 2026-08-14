import re
from pathlib import Path

from PIL import ExifTags, Image

# A flat photo has no scale reference on its own -- this only produces a
# real number when the image embeds altitude and focal length metadata
# (typical of drone photos; DJI stores altitude in XMP, not standard EXIF
# GPS tags). No metadata means no estimate: callers must not fabricate one.
SENSOR_WIDTH_35MM_MM = 36.0  # reference full-frame width; normalizes out
# per-camera sensor size differences when combined with the 35mm-equivalent
# focal length that most cameras already report in EXIF.


def _read_gps_altitude(image: Image.Image) -> float | None:
    exif = image.getexif()
    if not exif:
        return None
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if not gps_ifd:
        return None
    altitude = gps_ifd.get(6)  # GPSAltitude
    return float(altitude) if altitude is not None else None


def _read_xmp_altitude(image: Image.Image) -> float | None:
    """DJI and many other drone platforms store relative flight altitude in
    XMP metadata, not the standard EXIF GPS IFD."""
    xmp_bytes = image.info.get("xmp")
    if not xmp_bytes:
        return None
    xmp_text = xmp_bytes.decode("utf-8", errors="ignore")
    match = re.search(r'[Rr]elative[Aa]ltitude="?\+?(-?[\d.]+)', xmp_text)
    return float(match.group(1)) if match else None


def _read_focal_length_35mm(image: Image.Image) -> float | None:
    exif = image.getexif()
    if not exif:
        return None
    value = exif.get(ExifTags.Base.FocalLengthIn35mmFilm)
    return float(value) if value else None


def estimate_area_hectares(path: Path) -> tuple[float | None, str]:
    """Estimates the ground area covered by a photo from its embedded
    altitude and 35mm-equivalent focal length via the standard ground
    sample distance (GSD) formula: GSD = (sensor_width * altitude) /
    (focal_length * image_width_px). Returns (area_hectares, source) --
    area_hectares is None, with source "unavailable", whenever the needed
    metadata isn't present. That is the expected, common case (re-saved
    images, screenshots, non-drone cameras): callers must fall back to
    manual entry rather than guess.
    """
    try:
        with Image.open(path) as image:
            width_px, height_px = image.size
            altitude = _read_gps_altitude(image)
            source = "exif_gps_altitude"
            if altitude is None:
                altitude = _read_xmp_altitude(image)
                source = "xmp_relative_altitude"
            focal_length_35mm = _read_focal_length_35mm(image)
    except Exception:
        return None, "unavailable"

    if not altitude or altitude <= 0 or not focal_length_35mm or focal_length_35mm <= 0:
        return None, "unavailable"

    gsd_cm_per_px = (SENSOR_WIDTH_35MM_MM * altitude * 100) / (focal_length_35mm * width_px)
    area_m2 = (gsd_cm_per_px / 100 * width_px) * (gsd_cm_per_px / 100 * height_px)
    return round(area_m2 / 10000, 3), source
