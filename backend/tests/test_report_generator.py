import numpy as np
import pytest

from app.services.report_generator import generate_report_pdf
from app.services.schemas import AnalysisResult, Detection, PlantSizeStats


def make_result(**overrides) -> AnalysisResult:
    base = dict(
        plant_count=12,
        detection_method="fine_tuned",
        detection_note="12 plant(s) detected by the fine-tuned model.",
        crop_density=6.0,
        crop_coverage=54.2,
        vegetation_score=61.0,
        health_score=58.0,
        texture_uniformity_score=60.0,
        texture_pattern="mixed",
        tilt_corrected=False,
        tilt_correction_note="No correction needed.",
        estimated_yield=2.16,
        average_yield_per_plant_kg=0.18,
        confidence_score=71.4,
        detections=[],
        image_width=300,
        image_height=200,
        segmentation_overlay="data:image/png;base64,fake-seg",
        heatmap_overlay="data:image/png;base64,fake-heat",
    )
    base.update(overrides)
    return AnalysisResult(**base)


@pytest.fixture
def image() -> np.ndarray:
    return np.full((200, 300, 3), (60, 120, 60), dtype=np.uint8)


class TestGenerateReportPdf:
    def test_produces_a_valid_pdf(self, image):
        pdf_bytes = generate_report_pdf(
            image, make_result(), field_name="West Field", crop_type="Corn",
            field_area_hectares=3.2, analysis_date="Aug 15, 2026",
            health_label="Healthy vegetation", health_copy="Strong canopy.",
            recommendation="Keep monitoring.",
        )

        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 1000

    def test_includes_tilt_correction_and_plant_size_stats_when_present(self, image):
        # Regression test for a real gap: these two fields (Phases R/S)
        # were added to AnalysisResult but never wired into the PDF, so a
        # downloaded report silently omitted them even though the Results
        # screen showed both.
        result = make_result(
            tilt_corrected=True,
            tilt_correction_note="Perspective corrected using 40 row-direction and 12 cross-direction lines.",
            plant_size_stats=PlantSizeStats(
                plant_count=12, mean_area_cm2=850.5, median_area_cm2=800.0,
                min_area_cm2=400.0, max_area_cm2=1500.0, mean_aspect_ratio=1.4,
                size_uniformity_score=72.0,
            ),
        )

        pdf_bytes = generate_report_pdf(
            image, result, field_name="West Field", crop_type="Corn",
            field_area_hectares=3.2, analysis_date="Aug 15, 2026",
            health_label="Healthy vegetation", health_copy="Strong canopy.",
            recommendation="Keep monitoring.",
        )

        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 1000

    def test_includes_detections_with_boxes(self, image):
        result = make_result(detections=[
            Detection(x1=10, y1=10, x2=60, y2=60, confidence=0.8, label="crop"),
            Detection(x1=100, y1=80, x2=140, y2=120, confidence=0.5, label="weed"),
        ])

        pdf_bytes = generate_report_pdf(
            image, result, field_name="North Plot", crop_type="Wheat",
            field_area_hectares=1.5, analysis_date="Aug 15, 2026",
            health_label="Moderate / mixed health", health_copy="Some gaps.",
            recommendation="Inspect bare zones.",
        )

        assert pdf_bytes.startswith(b"%PDF")

    def test_handles_zero_detections_without_error(self, image):
        pdf_bytes = generate_report_pdf(
            image, make_result(plant_count=0, crop_density=0, confidence_score=0, detections=[]),
            field_name="Empty Field", crop_type="Rice", field_area_hectares=2.0,
            analysis_date="Aug 15, 2026", health_label="Poor or stressed vegetation",
            health_copy="Low coverage.", recommendation="Prioritize inspection.",
        )

        assert pdf_bytes.startswith(b"%PDF")
