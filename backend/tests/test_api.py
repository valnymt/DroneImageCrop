import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.schemas import AnalysisResult

client = TestClient(app)


def real_jpeg_bytes(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(60, 120, 60)).save(buf, format="JPEG")
    return buf.getvalue()

FAKE_RESULT = AnalysisResult(
    plant_count=42,
    crop_density=21.0,
    crop_coverage=55.5,
    vegetation_score=60.0,
    health_score=58.0,
    texture_uniformity_score=72.0,
    texture_pattern="uniform",
    estimated_yield=30.2,
    average_yield_per_plant_kg=0.02,
    confidence_score=88.0,
    detections=[],
    image_width=640,
    image_height=480,
    segmentation_overlay="data:image/png;base64,fake-seg",
    heatmap_overlay="data:image/png;base64,fake-heat",
)

# average_yield_per_plant_kg is intentionally omitted here -- it's now
# optional (see test_analyze_omits_yield_override_by_default below), and
# the "supply an override" path is exercised separately in
# test_analyze_passes_form_fields_through_to_pipeline.
VALID_FORM = {"crop_type": "Wheat", "field_size_hectares": "2"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "operational"}


def test_analyze_rejects_unsupported_content_type():
    response = client.post(
        "/api/analyze",
        files={"image": ("test.txt", b"not an image", "text/plain")},
        data=VALID_FORM,
    )
    assert response.status_code == 415


def test_analyze_requires_field_size_hectares():
    response = client.post(
        "/api/analyze",
        files={"image": ("test.jpg", b"fake-bytes", "image/jpeg")},
        data={"crop_type": "Wheat"},
    )
    assert response.status_code == 422


# The CV pipeline (YOLO/SAM/OpenCV) is mocked below -- these tests exercise
# the API contract (routing, validation, error mapping), not the models.


@patch("app.api.analysis.pipeline")
def test_analyze_returns_pipeline_result(mock_pipeline):
    mock_pipeline.analyze.return_value = FAKE_RESULT

    response = client.post(
        "/api/analyze",
        files={"image": ("test.jpg", b"fake-bytes", "image/jpeg")},
        data=VALID_FORM,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["plant_count"] == 42
    assert body["health_score"] == 58.0


@patch("app.api.analysis.pipeline")
def test_analyze_maps_value_error_to_422(mock_pipeline):
    mock_pipeline.analyze.side_effect = ValueError("The uploaded image could not be decoded.")

    response = client.post(
        "/api/analyze",
        files={"image": ("test.jpg", b"fake-bytes", "image/jpeg")},
        data=VALID_FORM,
    )

    assert response.status_code == 422
    assert "could not be decoded" in response.json()["detail"]


@patch("app.api.analysis.pipeline")
def test_analyze_passes_form_fields_through_to_pipeline(mock_pipeline):
    mock_pipeline.analyze.return_value = FAKE_RESULT

    client.post(
        "/api/analyze",
        files={"image": ("test.jpg", b"fake-bytes", "image/jpeg")},
        data={"crop_type": "Corn", "field_size_hectares": "3.5", "average_yield_per_plant_kg": "0.18"},
    )

    args, _ = mock_pipeline.analyze.call_args
    _, crop_type, field_size_hectares, average_yield_per_plant_kg = args
    assert crop_type == "Corn"
    assert field_size_hectares == 3.5
    assert average_yield_per_plant_kg == 0.18


@patch("app.api.analysis.pipeline")
def test_analyze_omits_yield_override_by_default(mock_pipeline):
    # No average_yield_per_plant_kg in the form (VALID_FORM doesn't include
    # it) -- the pipeline must receive None, not a fabricated frontend
    # default, so it resolves its own crop-specific baseline instead.
    mock_pipeline.analyze.return_value = FAKE_RESULT

    client.post(
        "/api/analyze",
        files={"image": ("test.jpg", b"fake-bytes", "image/jpeg")},
        data=VALID_FORM,
    )

    args, _ = mock_pipeline.analyze.call_args
    assert args[3] is None


@patch("app.api.analysis.pipeline")
def test_analyze_passes_settings_through_to_pipeline(mock_pipeline):
    mock_pipeline.analyze.return_value = FAKE_RESULT

    client.post(
        "/api/analyze",
        files={"image": ("test.jpg", b"fake-bytes", "image/jpeg")},
        data={**VALID_FORM, "enhance": "false", "refine_segmentation": "false", "conf_threshold": "0.4"},
    )

    _, kwargs = mock_pipeline.analyze.call_args
    assert kwargs["enhance"] is False
    assert kwargs["refine_segmentation"] is False
    assert kwargs["conf_threshold"] == 0.4


@patch("app.api.analysis.pipeline")
def test_analyze_settings_default_to_current_behavior(mock_pipeline):
    mock_pipeline.analyze.return_value = FAKE_RESULT

    client.post(
        "/api/analyze",
        files={"image": ("test.jpg", b"fake-bytes", "image/jpeg")},
        data=VALID_FORM,
    )

    _, kwargs = mock_pipeline.analyze.call_args
    assert kwargs["enhance"] is True
    assert kwargs["refine_segmentation"] is True
    assert kwargs["conf_threshold"] == 0.25


# /inspect (crop-type + area suggestions) -- CLIP and the area estimator are
# mocked below for the same reason: these tests exercise the API contract,
# not the model or the EXIF math (see test_field_area_estimator.py for that).


def test_inspect_rejects_unsupported_content_type():
    response = client.post(
        "/api/inspect",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


@patch("app.api.analysis.estimate_area_hectares")
@patch("app.api.analysis.classifier")
def test_inspect_returns_suggestions(mock_classifier, mock_estimate_area):
    mock_classifier.classify.return_value = ("Wheat", 0.6889)
    mock_estimate_area.return_value = (2.5, "exif_gps_altitude")

    response = client.post(
        "/api/inspect",
        files={"image": ("test.jpg", real_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["crop_type"] == "Wheat"
    assert body["confidence"] == 68.89
    assert body["estimated_area_hectares"] == 2.5
    assert body["area_source"] == "exif_gps_altitude"


@patch("app.api.analysis.estimate_area_hectares")
@patch("app.api.analysis.classifier")
def test_inspect_area_is_null_without_metadata(mock_classifier, mock_estimate_area):
    mock_classifier.classify.return_value = ("Tomato", 0.95)
    mock_estimate_area.return_value = (None, "unavailable")

    response = client.post(
        "/api/inspect",
        files={"image": ("test.jpg", real_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["estimated_area_hectares"] is None
    assert body["area_source"] == "unavailable"


@patch("app.api.analysis.estimate_area_hectares")
@patch("app.api.analysis.classifier")
def test_inspect_forwards_classified_crop_and_manual_altitude_to_area_estimator(mock_classifier, mock_estimate_area):
    # manual_altitude_m is the last-resort field the frontend only shows
    # after a first call comes back "unavailable" -- confirms it actually
    # reaches the estimator (as does the classifier's own crop guess, which
    # the row-spacing fallback needs to pick a realistic row spacing).
    mock_classifier.classify.return_value = ("Corn", 0.8)
    mock_estimate_area.return_value = (3.1, "manual_altitude")

    response = client.post(
        "/api/inspect",
        files={"image": ("test.jpg", real_jpeg_bytes(), "image/jpeg")},
        data={"manual_altitude_m": "75.0"},
    )

    assert response.status_code == 200
    assert response.json()["area_source"] == "manual_altitude"
    args, _ = mock_estimate_area.call_args
    assert args[1] == "Corn"
    assert args[2] == 75.0


@patch("app.api.analysis.estimate_area_hectares", return_value=(None, "unavailable"))
@patch("app.api.analysis.classifier")
def test_inspect_rejects_undecodable_image(mock_classifier, _mock_estimate_area):
    response = client.post(
        "/api/inspect",
        files={"image": ("test.jpg", b"not-a-real-jpeg", "image/jpeg")},
    )
    assert response.status_code == 422
    mock_classifier.classify.assert_not_called()


# /recompute (Results-screen crop/area override, Phase N) -- no image, no
# CV pipeline involved, so nothing here is mocked; it's real YieldEstimator
# math end-to-end through the HTTP layer.

RECOMPUTE_FORM = {
    "plant_count": "100", "crop_type": "Corn", "field_size_hectares": "2",
    "coverage": "80", "health": "70",
}


def test_recompute_returns_density_and_yield():
    response = client.post("/api/recompute", data=RECOMPUTE_FORM)

    assert response.status_code == 200
    body = response.json()
    assert body["crop_density"] == 50.0
    assert body["average_yield_per_plant_kg"] == 0.18
    condition_factor = 0.65 + 0.35 * ((80 + 70) / 200)
    assert body["estimated_yield"] == round(100 * 0.18 * condition_factor, 2)


def test_recompute_uses_corrected_crop_and_area():
    # Same plant_count/coverage/health as RECOMPUTE_FORM, but a different
    # crop (different baseline kg/plant) and a smaller area (higher
    # density) -- confirms both actually change the result, not just the
    # crop-density-only path.
    response = client.post(
        "/api/recompute",
        data={**RECOMPUTE_FORM, "crop_type": "Tomato", "field_size_hectares": "1"},
    )

    body = response.json()
    assert body["crop_density"] == 100.0
    assert body["average_yield_per_plant_kg"] == 3.0


def test_recompute_accepts_explicit_yield_override():
    response = client.post("/api/recompute", data={**RECOMPUTE_FORM, "average_yield_per_plant_kg": "0.5"})

    assert response.json()["average_yield_per_plant_kg"] == 0.5


def test_recompute_rejects_zero_area():
    response = client.post("/api/recompute", data={**RECOMPUTE_FORM, "field_size_hectares": "0"})

    assert response.status_code == 422


# /compare (flight-to-flight diff, Phase Q) -- FlightComparator is mocked
# below for the same reason as /recompute: these exercise the API
# contract, not the ORB/homography math (see test_flight_comparator.py for
# that).


def test_compare_rejects_unsupported_content_type():
    response = client.post(
        "/api/compare",
        files={
            "image_before": ("a.txt", b"not an image", "text/plain"),
            "image_after": ("b.jpg", real_jpeg_bytes(), "image/jpeg"),
        },
    )
    assert response.status_code == 415


@patch("app.api.analysis.comparator")
def test_compare_returns_alignment_result(mock_comparator):
    import numpy as np

    from app.services.flight_comparator import ComparisonResult

    mock_comparator.compare.return_value = ComparisonResult(
        alignment_ok=True, keypoints_matched=87, inlier_ratio=91.2,
        growth_percent=4.5, loss_percent=1.2, unchanged_percent=94.3,
        diff_overlay=np.zeros((10, 10, 3), dtype=np.uint8), warning=None,
    )

    response = client.post(
        "/api/compare",
        files={
            "image_before": ("before.jpg", real_jpeg_bytes(), "image/jpeg"),
            "image_after": ("after.jpg", real_jpeg_bytes(), "image/jpeg"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["alignment_ok"] is True
    assert body["keypoints_matched"] == 87
    assert body["growth_percent"] == 4.5
    assert body["diff_overlay"].startswith("data:image/png;base64,")
    assert body["warning"] is None


@patch("app.api.analysis.comparator")
def test_compare_surfaces_alignment_failure_without_erroring(mock_comparator):
    import numpy as np

    from app.services.flight_comparator import ComparisonResult

    mock_comparator.compare.return_value = ComparisonResult(
        alignment_ok=False, keypoints_matched=0, inlier_ratio=0.0,
        growth_percent=0.0, loss_percent=0.0, unchanged_percent=0.0,
        diff_overlay=np.zeros((1, 1, 3), dtype=np.uint8),
        warning="Only 3 reliable matching features found between the two photos (need at least 12).",
    )

    response = client.post(
        "/api/compare",
        files={
            "image_before": ("before.jpg", real_jpeg_bytes(), "image/jpeg"),
            "image_after": ("after.jpg", real_jpeg_bytes(), "image/jpeg"),
        },
    )

    assert response.status_code == 200  # a failed alignment is a valid, honest response -- not a server error
    body = response.json()
    assert body["alignment_ok"] is False
    assert "reliable matching features" in body["warning"]


def test_compare_rejects_undecodable_image():
    response = client.post(
        "/api/compare",
        files={
            "image_before": ("before.jpg", b"not-a-real-jpeg", "image/jpeg"),
            "image_after": ("after.jpg", real_jpeg_bytes(), "image/jpeg"),
        },
    )
    assert response.status_code == 422


# /report (PDF export) -- generate_report_pdf is mocked below for the same
# reason: these exercise the API contract, not PDF rendering (see
# test_report_generator.py for that).

REPORT_FORM = {
    "result": FAKE_RESULT.model_dump_json(),
    "field_name": "West Field",
    "crop_type": "Wheat",
    "field_area_hectares": "2",
    "analysis_date": "Aug 15, 2026",
    "health_label": "Healthy vegetation",
    "health_copy": "Strong canopy.",
    "recommendation": "Keep monitoring.",
}


def test_report_rejects_unsupported_content_type():
    response = client.post(
        "/api/report",
        files={"image": ("test.txt", b"not an image", "text/plain")},
        data=REPORT_FORM,
    )
    assert response.status_code == 415


def test_report_rejects_malformed_result_json():
    response = client.post(
        "/api/report",
        files={"image": ("test.jpg", real_jpeg_bytes(), "image/jpeg")},
        data={**REPORT_FORM, "result": "not json"},
    )
    assert response.status_code == 422


def test_report_rejects_undecodable_image():
    response = client.post(
        "/api/report",
        files={"image": ("test.jpg", b"not-a-real-jpeg", "image/jpeg")},
        data=REPORT_FORM,
    )
    assert response.status_code == 422


@patch("app.api.analysis.generate_report_pdf")
def test_report_returns_pdf_with_safe_filename(mock_generate):
    mock_generate.return_value = b"%PDF-fake-bytes"

    response = client.post(
        "/api/report",
        files={"image": ("test.jpg", real_jpeg_bytes(), "image/jpeg")},
        data={**REPORT_FORM, "field_name": 'West "Field" / Plot #2'},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"%PDF-fake-bytes"
    disposition = response.headers["content-disposition"]
    filename = disposition.split('filename="', 1)[1].rsplit('"', 1)[0]
    # The raw field_name has quotes and a slash -- confirms _safe_pdf_filename
    # actually stripped them rather than passing user input straight into
    # the header (a header-injection / malformed-header risk otherwise).
    assert '"' not in filename
    assert "/" not in filename
    assert filename.endswith(".pdf")
    assert "West" in filename and "Field" in filename and "Plot" in filename


@patch("app.api.analysis.generate_report_pdf")
def test_report_passes_parsed_result_through(mock_generate):
    mock_generate.return_value = b"%PDF-fake-bytes"

    client.post(
        "/api/report",
        files={"image": ("test.jpg", real_jpeg_bytes(), "image/jpeg")},
        data=REPORT_FORM,
    )

    args, _ = mock_generate.call_args
    _, analysis_result, field_name, crop_type, field_area_hectares, analysis_date, health_label, health_copy, recommendation = args
    assert analysis_result.plant_count == FAKE_RESULT.plant_count
    assert field_name == "West Field"
    assert crop_type == "Wheat"
    assert field_area_hectares == 2.0
    assert analysis_date == "Aug 15, 2026"
    assert health_label == "Healthy vegetation"
    assert recommendation == "Keep monitoring."
