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


@patch("app.api.analysis.estimate_area_hectares", return_value=(None, "unavailable"))
@patch("app.api.analysis.classifier")
def test_inspect_rejects_undecodable_image(mock_classifier, _mock_estimate_area):
    response = client.post(
        "/api/inspect",
        files={"image": ("test.jpg", b"not-a-real-jpeg", "image/jpeg")},
    )
    assert response.status_code == 422
    mock_classifier.classify.assert_not_called()
