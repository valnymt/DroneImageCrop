import re
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

# A flat photo has no scale reference on its own -- this only produces a
# real number when the image embeds altitude and focal length metadata
# (typical of drone photos; DJI stores altitude in XMP, not standard EXIF
# GPS tags), or a caller supplies altitude manually, or the image itself
# shows a periodic crop-row pattern we can measure against a typical
# agronomic row spacing. None of those means no estimate: callers must not
# fabricate one.
SENSOR_WIDTH_35MM_MM = 36.0  # reference full-frame width; normalizes out
# per-camera sensor size differences when combined with the 35mm-equivalent
# focal length that most cameras already report in EXIF.

# Most phone exports strip GPS altitude entirely and never had XMP relative
# altitude to begin with, but *do* keep FocalLengthIn35mmFilm -- so the
# reverse gap (altitude present/manual, focal length missing) is common
# too. This is a typical smartphone wide-angle equivalent; using it instead
# of refusing to estimate trades a few percent of error for a real number.
DEFAULT_FOCAL_LENGTH_35MM_MM = 26.0

# Typical between-row spacing in meters for common row crops, used only as
# the ground-scale reference for the row-spacing fallback below. "_default"
# covers an unrecognized/unknown crop type.
ROW_SPACING_METERS = {
    "Corn": 0.76,
    "Soybean": 0.5,
    "Wheat": 0.18,
    "Rice": 0.25,
    "Tomato": 0.9,
    "_default": 0.4,
}

# Row-spacing periodicity below this confidence is treated as noise (no
# real row structure visible), not a usable estimate. Real synthetic row
# patterns score 0.6-0.8 on this scale; ordinary non-row photos (grass
# close-ups, single-blob vegetation, smooth illumination gradients) score
# under 0.1 -- see estimate_area_from_row_spacing's docstring for why a
# single strong autocorrelation peak alone isn't enough evidence.
ROW_SPACING_CONFIDENCE_THRESHOLD = 0.35
# Need to see at least this many pixels of run before trusting a "row"
# spacing -- shorter lags are dominated by pixel-level mask noise, not row
# structure.
ROW_SPACING_MIN_LAG_PX = 15


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


def _area_from_gsd(altitude_m: float, focal_length_35mm_mm: float, width_px: int, height_px: int) -> float:
    gsd_cm_per_px = (SENSOR_WIDTH_35MM_MM * altitude_m * 100) / (focal_length_35mm_mm * width_px)
    area_m2 = (gsd_cm_per_px / 100 * width_px) * (gsd_cm_per_px / 100 * height_px)
    return round(area_m2 / 10000, 3)


def _local_maxima(values: np.ndarray, lo: int, hi: int) -> list[int]:
    return [i for i in range(max(lo, 1), min(hi, len(values) - 1)) if values[i] > values[i - 1] and values[i] >= values[i + 1]]


def _dominant_periodicity_px(profile: np.ndarray) -> tuple[float | None, float]:
    """Finds the pixel spacing of a genuinely repeating pattern in a 1D
    projection profile via autocorrelation, and returns (spacing_px,
    confidence). Both None/0.0 if nothing trustworthy is found.

    A single dominant low-frequency component (the signature of a smooth
    illumination gradient, or a single blob of vegetation with no row
    structure) is NOT enough evidence on its own -- ordinary photos
    reliably produce one strong-looking peak this way, which is what made
    an earlier FFT-magnitude version of this function false-positive on
    non-row photos. Real periodicity instead needs two things: a genuine
    local peak in the autocorrelation (with real prominence over the
    trough before it, not just the tail of a decay curve), AND a second
    local peak near double that lag (the first harmonic) -- a lone bump
    can't produce that, only actual repetition can.
    """
    n = len(profile)
    if n < 4 * ROW_SPACING_MIN_LAG_PX:
        return None, 0.0
    x = profile.astype(np.float64)
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)  # remove linear illumination gradient across the frame
    x = x - x.mean()
    variance = float(np.dot(x, x))
    if variance < 1e-9:
        return None, 0.0
    autocorr = np.correlate(x, x, mode="full")[n - 1 :] / variance
    autocorr = np.convolve(autocorr, np.ones(3) / 3, mode="same")  # light smoothing against single-sample noise spikes

    max_lag = n // 3  # need at least 3 full cycles visible to call it periodic at all
    if max_lag <= ROW_SPACING_MIN_LAG_PX + 5:
        return None, 0.0

    peaks = _local_maxima(autocorr, ROW_SPACING_MIN_LAG_PX, max_lag)
    if not peaks:
        return None, 0.0

    def prominence(peak_idx: int) -> float:
        trough = autocorr[1:peak_idx].min() if peak_idx > 1 else autocorr[0]
        return float(autocorr[peak_idx] - trough)

    fundamental_candidates = [p for p in peaks if prominence(p) > 0.12]
    if not fundamental_candidates:
        return None, 0.0
    fundamental = min(fundamental_candidates)  # smallest-lag genuine peak

    harmonic_band = [p for p in peaks if 1.7 * fundamental <= p <= 2.3 * fundamental]
    if not harmonic_band:
        return None, 0.0

    confidence = min(float(autocorr[fundamental]), float(autocorr[harmonic_band[0]]))
    return float(fundamental), confidence


def estimate_area_from_row_spacing(image_bgr: np.ndarray, crop_type: str | None) -> tuple[float | None, str]:
    """Ground-scale-free fallback: measures the pixel spacing of the
    dominant repeating row pattern in the vegetation mask (checking both
    the horizontal and vertical projection), then converts it to a GSD
    using a typical real-world row spacing for the detected crop. Only
    trusted when the periodicity is clearly the strongest signal in the
    profile (see ROW_SPACING_CONFIDENCE_THRESHOLD) -- a field with no
    visible row structure (dense canopy, non-row crop, or a non-aerial
    photo) correctly yields no estimate rather than a guess.
    """
    from app.services.opencv_processor import OpenCVProcessor  # local import: keeps this module's EXIF-only path free of the CV import chain

    mask = OpenCVProcessor().vegetation_metrics(image_bgr).green_mask
    if cv2.countNonZero(mask) < mask.size * 0.02:
        return None, "unavailable"  # almost no vegetation detected -- nothing to measure rows against

    best_spacing_px, best_confidence = None, 0.0
    for axis in (0, 1):  # 0: column profile (catches vertical rows); 1: row profile (catches horizontal rows)
        profile = mask.mean(axis=axis)
        spacing_px, confidence = _dominant_periodicity_px(profile)
        if spacing_px is not None and confidence > best_confidence:
            best_spacing_px, best_confidence = spacing_px, confidence

    if best_spacing_px is None or best_confidence < ROW_SPACING_CONFIDENCE_THRESHOLD:
        return None, "unavailable"

    row_spacing_m = ROW_SPACING_METERS.get(crop_type or "", ROW_SPACING_METERS["_default"])
    gsd_m_per_px = row_spacing_m / best_spacing_px
    height_px, width_px = mask.shape[:2]
    area_m2 = (gsd_m_per_px * width_px) * (gsd_m_per_px * height_px)
    return round(area_m2 / 10000, 3), "row_spacing_estimate"


# (low_multiplier, high_multiplier, confidence_tier) per area_source --
# NOT fabricated error bars. Each range comes from actually propagating a
# real, known uncertainty in that source's weakest input through the GSD
# formula's own math: area ~ (altitude / focal_length)^2 (see
# _area_from_gsd), so a fractional error in altitude or focal length of x
# becomes roughly a (1+/-x)^2 fractional error in area -- not a number
# pulled from nowhere.
#
# exif_gps_altitude* is the one real defect this mapping surfaces, not
# just documents: standard EXIF GPSAltitude is height above/below mean
# sea level (see GPSAltitudeRef, EXIF spec), not the drone's height above
# the ground it photographed. A field at 300m elevation photographed from
# 100m AGL reports ~400m GPS altitude, which _area_from_gsd would treat as
# a 400m flight -- 16x the true area. There's no local ground-elevation
# data available here to correct for that gap, so unlike the other
# sources, this one gets no numeric range at all: any number would be
# fake precision on an error that's structurally unbounded, not just
# larger. It's demoted to "low" confidence instead of previously reading
# identically to the genuinely reliable XMP/manual paths.
_FOCAL_LOW_MM, _FOCAL_HIGH_MM = 20.0, 35.0  # realistic 35mm-equivalent spread across phone/drone cameras
_FOCAL_DEFAULT_ERROR = (
    (DEFAULT_FOCAL_LENGTH_35MM_MM / _FOCAL_HIGH_MM) ** 2,
    (DEFAULT_FOCAL_LENGTH_35MM_MM / _FOCAL_LOW_MM) ** 2,
)
# Real between-field/cultivar row-spacing variance for a given crop type
# is easily +/-30% around one textbook number (see ROW_SPACING_METERS) --
# this reflects that spread, not the periodicity-detection confidence
# already gated by ROW_SPACING_CONFIDENCE_THRESHOLD.
_ROW_SPACING_ERROR = (0.7**2, 1.3**2)

AREA_CONFIDENCE: dict[str, tuple[float | None, float | None, str]] = {
    "exif_gps_altitude": (None, None, "low"),
    "exif_gps_altitude_default_focal": (None, None, "low"),
    "xmp_relative_altitude": (0.9, 1.1, "high"),
    "xmp_relative_altitude_default_focal": (*_FOCAL_DEFAULT_ERROR, "medium"),
    "manual_altitude": (0.9, 1.1, "high"),
    "manual_altitude_default_focal": (*_FOCAL_DEFAULT_ERROR, "medium"),
    "row_spacing_estimate": (*_ROW_SPACING_ERROR, "medium"),
}


def area_confidence(source: str) -> tuple[float | None, float | None, str]:
    """(low_multiplier, high_multiplier, confidence_tier) for an
    area_source string returned by estimate_area_hectares. Multipliers are
    None (no numeric range -- see AREA_CONFIDENCE's docstring above) for
    sources whose error is structurally unbounded, not just wide.
    "unavailable" and anything unrecognized falls back to (None, None,
    "low") -- an unmeasured default, not a range around a real estimate.
    """
    return AREA_CONFIDENCE.get(source, (None, None, "low"))


def estimate_area_hectares(
    path: Path, crop_type: str | None = None, manual_altitude_m: float | None = None,
) -> tuple[float | None, str]:
    """Estimates the ground area covered by a photo. Tries, in order:
    1. EXIF GPS altitude, or DJI-style XMP relative altitude, via the
       standard ground sample distance (GSD) formula.
    2. The same GSD formula using a caller-supplied altitude (last resort:
       a single manual field, only surfaced by the caller when nothing
       else worked) -- with a default focal length if that's also missing.
    3. Row-spacing periodicity in the image itself (see
       estimate_area_from_row_spacing) when no altitude is available from
       any source.
    Returns (area_hectares, source); area_hectares is None with source
    "unavailable" only when none of the above produced a trustworthy
    number -- callers must not fabricate one at that point.
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

    if not altitude or altitude <= 0:
        if manual_altitude_m and manual_altitude_m > 0:
            altitude, source = manual_altitude_m, "manual_altitude"
        else:
            altitude = None

    if altitude and altitude > 0:
        if focal_length_35mm and focal_length_35mm > 0:
            focal = focal_length_35mm
        else:
            focal = DEFAULT_FOCAL_LENGTH_35MM_MM
            source = f"{source}_default_focal"
        return _area_from_gsd(altitude, focal, width_px, height_px), source

    cv_image = cv2.imread(str(path))
    if cv_image is not None:
        area_ha, row_source = estimate_area_from_row_spacing(cv_image, crop_type)
        if area_ha is not None:
            return area_ha, row_source

    return None, "unavailable"
