import pytest
from pydantic import ValidationError

from app.services.schemas import AnalysisResult, Detection


def make_result(**overrides):
    base = dict(
        plant_count=10,
        crop_density=5.0,
        crop_coverage=50.0,
        vegetation_score=60.0,
        health_score=70.0,
        estimated_yield=12.5,
        average_yield_per_plant_kg=0.02,
        confidence_score=80.0,
        detections=[],
        image_width=640,
        image_height=480,
        segmentation_overlay="data:image/png;base64,fake-seg",
        heatmap_overlay="data:image/png;base64,fake-heat",
    )
    base.update(overrides)
    return AnalysisResult(**base)


class TestAnalysisResult:
    def test_valid_payload_constructs(self):
        result = make_result()
        assert result.plant_count == 10

    @pytest.mark.parametrize("field", ["crop_coverage", "vegetation_score", "health_score", "confidence_score"])
    def test_percentage_fields_reject_values_over_100(self, field):
        with pytest.raises(ValidationError):
            make_result(**{field: 150.0})

    @pytest.mark.parametrize(
        "field", ["plant_count", "crop_density", "estimated_yield", "average_yield_per_plant_kg", "confidence_score"],
    )
    def test_nonnegative_fields_reject_negative_values(self, field):
        with pytest.raises(ValidationError):
            make_result(**{field: -1})

    @pytest.mark.parametrize("field", ["image_width", "image_height"])
    def test_image_dimensions_reject_nonpositive_values(self, field):
        with pytest.raises(ValidationError):
            make_result(**{field: 0})

    def test_detections_default_to_a_list(self):
        detection = Detection(x1=1.0, y1=2.0, x2=30.0, y2=40.0, confidence=0.92, label="crop")
        result = make_result(detections=[detection])
        assert result.detections[0].label == "crop"
        assert result.detections[0].confidence == pytest.approx(0.92)

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisResult(plant_count=1, crop_density=1.0, crop_coverage=1.0, vegetation_score=1.0, detections=[])
