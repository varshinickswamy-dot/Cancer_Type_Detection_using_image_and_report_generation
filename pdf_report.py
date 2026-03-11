from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
import os
import uuid


def get_treatment(cancer_type, result):

    treatments = {
        "breast": {
            "benign": "Regular monitoring, ultrasound follow-up, lifestyle improvement.",
            "malignant": "Surgery, Chemotherapy, Radiation therapy, Hormone therapy."
        },
        "lung": {
            "normal": "No treatment required. Maintain healthy lifestyle.",
            "abnormal": "CT scan evaluation, Biopsy, Chemotherapy, Radiation therapy."
        },
        "skin": {
            "normal": "No treatment required. Maintain skincare hygiene.",
            "affected": "Dermatologist consultation, Surgical removal, Cryotherapy."
        }
    }

    return treatments.get(cancer_type, {}).get(result, "Consult specialist.")


def generate_report(path, cancer_type, result, confidence, image_path=None):

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    elements = []
    styles = getSampleStyleSheet()

    # =========================
    # HEADER
    # =========================

    header_style = styles["Title"]
    header_style.alignment = 1

    elements.append(Paragraph("AI Multi-Speciality Diagnostic Center", header_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Cancer Detection Report", styles["Heading2"]))
    elements.append(Spacer(1, 20))

    # =========================
    # REPORT DETAILS TABLE
    # =========================

    report_id = str(uuid.uuid4())[:8]
    date_time = datetime.now().strftime("%d-%m-%Y %H:%M")

    details_data = [
        ["Report ID:", report_id],
        ["Date:", date_time],
        ["Cancer Type:", cancer_type.capitalize()],
        ["Prediction:", result.capitalize()],
        ["Confidence:", f"{round(confidence,2)} %"]
    ]

    table = Table(details_data, colWidths=[120, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    # =========================
    # IMAGE SECTION
    # =========================

    if image_path and os.path.exists(image_path):
        elements.append(Paragraph("Scanned Image:", styles["Heading3"]))
        elements.append(Spacer(1, 10))

        img = Image(image_path, width=3*inch, height=3*inch)
        img.hAlign = "CENTER"
        elements.append(img)
        elements.append(Spacer(1, 20))

    # =========================
    # RESULT HIGHLIGHT BOX
    # =========================

    box_color = colors.green if result.lower() in ["benign", "normal"] else colors.red

    result_table = Table(
        [[f"Final Diagnosis: {result.upper()}"]],
        colWidths=[420]
    )

    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), box_color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
    ]))

    elements.append(result_table)
    elements.append(Spacer(1, 20))

    # =========================
    # TREATMENT SECTION
    # =========================

    elements.append(Paragraph("Recommended Treatment / Next Steps:", styles["Heading3"]))
    elements.append(Spacer(1, 10))

    treatment_text = get_treatment(cancer_type, result)
    elements.append(Paragraph(treatment_text, styles["Normal"]))
    elements.append(Spacer(1, 30))

    # =========================
    # FOOTER
    # =========================

    elements.append(Paragraph(
        "Note: This is an AI-generated preliminary report. "
        "Please consult a certified medical professional for confirmation.",
        styles["Italic"]
    ))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Authorized Signature: ____________________", styles["Normal"]))

    doc.build(elements)