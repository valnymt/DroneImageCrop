import io

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.services.schemas import AnalysisResult

BRAND_GREEN = colors.HexColor("#174d32")
INK = colors.HexColor("#17241c")
MUTED = colors.HexColor("#68736b")
LINE = colors.HexColor("#dde2da")

# Mirrors the fixed disclaimer text already shown in the Results UI
# (app/page.tsx) -- kept here rather than passed from the frontend since
# it's static copy, not something derived per-analysis.
RGB_SCREENING_DISCLAIMER = (
    "Color analysis can flag suspicious areas, but cannot distinguish disease from "
    "drought, mature crops, harvest residue, shadows, or soil without field context."
)
TEXTURE_DISCLAIMER = (
    "Texture pattern (GLCM/Haralick features on the segmented canopy) separates uniform "
    "condition changes, e.g. drought or nutrient stress, from patchy ones, e.g. disease or "
    "pest damage -- it is not a diagnosis of which specific condition is present."
)


def _annotate_detections(image: np.ndarray, result: AnalysisResult) -> io.BytesIO:
    """Draws the real detection boxes onto the analyzed image for the PDF --
    unlike the frontend's results view, this uses the actual x1/y1/x2/y2
    coordinates rather than placeholder positions."""
    annotated = image.copy()
    for detection in result.detections:
        p1 = (int(detection.x1), int(detection.y1))
        p2 = (int(detection.x2), int(detection.y2))
        cv2.rectangle(annotated, p1, p2, (60, 200, 90), 2)
        label = f"{detection.label} {detection.confidence:.0%}"
        text_origin = (p1[0], max(p1[1] - 6, 12))
        cv2.putText(annotated, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 200, 90), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise ValueError("Could not encode the annotated image.")
    return io.BytesIO(buf.tobytes())


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], textColor=BRAND_GREEN, fontName="Helvetica-Bold"),
        "heading": ParagraphStyle("ReportHeading", parent=base["Heading2"], textColor=BRAND_GREEN, spaceBefore=4),
        "body": ParagraphStyle("ReportBody", parent=base["BodyText"], textColor=INK, leading=14),
        "muted": ParagraphStyle("ReportMuted", parent=base["BodyText"], textColor=MUTED, fontSize=9, leading=12),
    }


def generate_report_pdf(
    image: np.ndarray,
    result: AnalysisResult,
    field_name: str,
    crop_type: str,
    field_area_hectares: float,
    analysis_date: str,
    health_label: str,
    health_copy: str,
    recommendation: str,
) -> bytes:
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )

    story = [
        Paragraph("AgriSight Field Analysis Report", styles["title"]),
        Paragraph(f"{field_name} · {crop_type} · {analysis_date}", styles["muted"]),
        Spacer(1, 0.25 * inch),
    ]

    meta_table = Table(
        [
            ["Field name", field_name, "Crop type", crop_type],
            ["Field area", f"{field_area_hectares:g} ha", "Analysis date", analysis_date],
        ],
        colWidths=[1.1 * inch, 2.1 * inch, 1.1 * inch, 2.1 * inch],
    )
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
    ]))
    story += [meta_table, Spacer(1, 0.25 * inch), Paragraph("Headline metrics", styles["heading"])]

    metrics_table = Table(
        [
            ["Plant count", "Crop density", "Crop coverage", "Health score", "Est. harvest"],
            [
                f"{result.plant_count:,}",
                f"{result.crop_density:,.2f} / ha",
                f"{result.crop_coverage:.1f}%",
                f"{result.health_score:.0f} / 100",
                f"{result.estimated_yield:,.2f} kg",
            ],
        ],
        colWidths=[1.08 * inch] * 5,
    )
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
    ]))
    story += [metrics_table, Spacer(1, 0.3 * inch), Paragraph("Computer vision output", styles["heading"])]

    h, w = image.shape[:2]
    display_width = 6.3 * inch
    display_height = display_width * h / w
    max_height = 4.2 * inch
    if display_height > max_height:
        display_height = max_height
        display_width = display_height * w / h
    story.append(RLImage(_annotate_detections(image, result), width=display_width, height=display_height))
    story.append(Paragraph(
        f"{len(result.detections)} YOLO detection(s) shown · "
        f"avg. detection confidence {result.confidence_score:.1f}%",
        styles["muted"],
    ))
    story += [Spacer(1, 0.3 * inch), Paragraph("Field intelligence", styles["heading"])]

    story += [
        Paragraph(f"<b>{health_label}</b>", styles["body"]),
        Paragraph(health_copy, styles["body"]),
        Spacer(1, 0.08 * inch),
        Paragraph(f"Green vegetation ratio: {result.vegetation_score:.1f}%", styles["body"]),
        Paragraph(
            f"Texture pattern: {result.texture_pattern} ({result.texture_uniformity_score:.0f}/100 uniformity)",
            styles["body"],
        ),
    ]
    if result.tilt_corrected:
        story.append(Paragraph(f"Perspective correction: {result.tilt_correction_note}", styles["body"]))
    if result.plant_size_stats is not None:
        stats = result.plant_size_stats
        story += [
            Spacer(1, 0.08 * inch),
            Paragraph(
                f"Per-plant size: {stats.mean_area_cm2:,.0f} cm² mean canopy area "
                f"({stats.min_area_cm2:,.0f}-{stats.max_area_cm2:,.0f} cm² range), "
                f"{stats.size_uniformity_score:.0f}/100 size uniformity, "
                f"{stats.mean_aspect_ratio:.2f}x mean elongation",
                styles["body"],
            ),
        ]
    story += [
        Spacer(1, 0.15 * inch),
        Paragraph("<b>RGB screening result</b>", styles["body"]),
        Paragraph(RGB_SCREENING_DISCLAIMER, styles["muted"]),
        Paragraph(TEXTURE_DISCLAIMER, styles["muted"]),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Recommendation</b>", styles["body"]),
        Paragraph(recommendation, styles["body"]),
    ]

    doc.build(story)
    return buffer.getvalue()
