
import os
import uuid
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    HRFlowable
)


def create_prediction_report(
    output_path,
    disease,
    confidence,
    original_image,
    heatmap_image,
    overlay_image,
    disease_details
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        textColor=colors.HexColor("#176b3a"),
        alignment=TA_CENTER,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#176b3a"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=6
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=body_style,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#765d20"),
        backColor=colors.HexColor("#fff8e6"),
        borderPadding=9
    )

    story = []

    # Generate unique report ID
    report_id = (
        "PCAI-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:8].upper()
    )

    # Header
    story.append(Paragraph("PLANTCARE AI", title_style))

    story.append(
        Paragraph("Plant Disease Detection Report", subtitle_style)
    )

    story.append(
        HRFlowable(
            width="100%",
            color=colors.HexColor("#21864b"),
            thickness=1.5
        )
    )

    story.append(Spacer(1, 15))

    # Report information
    report_date = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")

    report_info = Table(
        [
            ["Report ID", report_id],
            ["Model Name", "ResNet18"],
            ["Report Generated", report_date]
        ],
        colWidths=[160, 300]
    )

    report_info.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1),
             colors.HexColor("#e5f4e9")),
            ("TEXTCOLOR", (0, 0), (0, -1),
             colors.HexColor("#176b3a")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
        ])
    )

    story.append(report_info)

    # Prediction details
    story.append(
        Paragraph("Prediction Result", heading_style)
    )

    disease_name = (
        str(disease)
        .replace("___", " - ")
        .replace("_", " ")
    )

    result_table = Table(
        [
            ["Predicted Condition", disease_name],
            ["Confidence", f"{confidence}%"]
        ],
        colWidths=[160, 300]
    )

    result_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1),
             colors.HexColor("#e5f4e9")),
            ("TEXTCOLOR", (0, 0), (0, -1),
             colors.HexColor("#176b3a")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
        ])
    )

    story.append(result_table)

    # Original image
    story.append(
        Paragraph("Original Leaf Image", heading_style)
    )

    if original_image and os.path.exists(original_image):

        img = PDFImage(
            original_image,
            width=3.2 * inch,
            height=2.5 * inch,
            kind="proportional"
        )

        img.hAlign = "CENTER"
        story.append(img)

    # Grad-CAM images
    story.append(
        Paragraph("Grad-CAM Visualization", heading_style)
    )

    visualization_data = []

    heatmap_cell = Paragraph("Heatmap unavailable", body_style)
    overlay_cell = Paragraph("Overlay unavailable", body_style)

    if heatmap_image and os.path.exists(heatmap_image):
        heatmap_cell = PDFImage(
            heatmap_image,
            width=2.4 * inch,
            height=1.8 * inch,
            kind="proportional"
        )

    if overlay_image and os.path.exists(overlay_image):
        overlay_cell = PDFImage(
            overlay_image,
            width=2.4 * inch,
            height=1.8 * inch,
            kind="proportional"
        )

    visualization_data.append([heatmap_cell, overlay_cell])

    visualization_data.append([
        Paragraph("Grad-CAM Heatmap", subtitle_style),
        Paragraph("Grad-CAM Overlay", subtitle_style)
    ])

    visualization_table = Table(
        visualization_data,
        colWidths=[3.0 * inch, 3.0 * inch]
    )

    visualization_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
        ])
    )

    story.append(visualization_table)

    story.append(
        Paragraph(
            "Grad-CAM highlights image regions that influenced "
            "the model prediction. It does not independently confirm "
            "disease location or severity.",
            body_style
        )
    )

    # Disease description
    story.append(
        Paragraph("Disease Information", heading_style)
    )

    description = disease_details.get(
        "description",
        "No additional information available."
    )

    story.append(
        Paragraph(str(description), body_style)
    )

    # Disease information sections
    sections = [
        ("Symptoms", "symptoms"),
        ("Causes", "causes"),
        ("Prevention", "prevention"),
        ("Management and Treatment", "management")
    ]

    for heading, key in sections:

        story.append(
            Paragraph(heading, heading_style)
        )

        items = disease_details.get(key, [])

        if isinstance(items, str):
            items = [items]

        if items:
            for item in items:
                story.append(
                    Paragraph(
                        "&#8226; " + str(item),
                        body_style
                    )
                )
        else:
            story.append(
                Paragraph("Information not available.", body_style)
            )

    # Disclaimer
    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "<b>Important Disclaimer:</b> This report is generated "
            "using an AI-based prediction system for educational "
            "purposes. Predictions are not confirmed agricultural "
            "diagnoses. Consult an agricultural expert before "
            "applying treatments.",
            disclaimer_style
        )
    )

    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "PlantCare AI | Deep Learning Based Plant Disease Detection",
            subtitle_style
        )
    )

    # Generate PDF
    document.build(story)

    return output_path
