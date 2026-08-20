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
    estimated_yield=30.2,
    confidence_score=88.0,
    detections=[],
    image_width=640,
    image_height=480,
    segmentation_overlay="data:image/png;base64,fake-seg",
    heatmap_overlay="data:image/png;base64,fake-heat",
)

VALID_FORM = {"crop_type": "Wheat", "field_size_hectares": "2", "average_yield_per_plant_kg": "0.02"}


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
