import reportlab
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import pagesizes
from datetime import datetime


# -----------------------------
# Convert findings → sentences
# -----------------------------
def generate_medical_text(findings):

    text = ""

    if findings["Liver"] == "Abnormal":
        text += "The liver shows features suggestive of fatty changes. "
    else:
        text += "The liver appears normal in size and echotexture. "

    if findings["Gall Bladder"] == "Abnormal":
        text += "The gall bladder shows abnormal findings. "
    else:
        text += "The gall bladder appears normal with no calculi. "

    if findings["Kidneys"] == "Abnormal":
        text += "Kidneys show abnormal features. "
    else:
        text += "Both kidneys are normal in size and echotexture. "

    if findings["Free Fluid"] == "Present":
        text += "Free fluid is noted in the abdomen. "
    else:
        text += "No free fluid is seen in the abdomen. "

    return text


# -----------------------------
# Generate PDF
# -----------------------------
def generate_pdf(findings, patient_name, age, gender, doctor_name, transcription, filename="ultrasound_report.pdf"):

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'title',
        parent=styles['Title'],
        alignment=1,
        fontSize=16
    )

    section_style = ParagraphStyle(
        'section',
        parent=styles['Heading2'],
        fontSize=12
    )

    doc = SimpleDocTemplate(
        filename,
        pagesize=pagesizes.A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=40,
        bottomMargin=30
    )

    elements = []

    # -----------------------------
    # Header (Hospital Style)
    # -----------------------------
    elements.append(Paragraph("<b>AL SHIFA DIAGNOSTIC CENTER</b>", title_style))
    elements.append(Paragraph("Malappuram, Kerala | Ph: +91-XXXXXXXXXX", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(HRFlowable(width="100%", thickness=1.5))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>ULTRASOUND ABDOMEN REPORT</b>", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # -----------------------------
    # Patient Info
    # -----------------------------
    patient_table = [
    ["Patient Name", str(patient_name if patient_name else "N/A"), "Date", datetime.now().strftime("%d-%m-%Y")],
    ["Age / Gender", f"{age} / {gender}", "Doctor", doctor_name],
]

    table = Table(patient_table, colWidths=[100, 150, 100, 150])
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # -----------------------------
    # FINDINGS
    # -----------------------------
    elements.append(Paragraph("<b>FINDINGS:</b>", section_style))
    elements.append(Spacer(1, 0.1 * inch))

    full_report = generate_medical_text(findings) + "<br/><br/><b>Doctor Notes:</b><br/>" + transcription
    elements.append(Paragraph(full_report, styles["Normal"]))
    
    elements.append(Spacer(1, 0.3 * inch))

    # -----------------------------
    # IMPRESSION
    # -----------------------------
    elements.append(Paragraph("<b>IMPRESSION:</b>", section_style))
    elements.append(Spacer(1, 0.1 * inch))

    if findings["Liver"] == "Abnormal":
        impression = "Features suggestive of fatty liver."
    else:
        impression = "No significant abnormality detected."

    elements.append(Paragraph(impression, styles["Normal"]))
    elements.append(Spacer(1, 0.4 * inch))

    # -----------------------------
    # IMAGE PLACEHOLDER
    # -----------------------------
    img_table = Table([[" Ultrasound Image Placeholder "]], colWidths=[450], rowHeights=[140])
    img_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))

    elements.append(img_table)
    elements.append(Spacer(1, 0.4 * inch))

    # -----------------------------
    # DOCTOR REMARKS (NEW)
    # -----------------------------
    elements.append(Paragraph("<b>DOCTOR REMARKS:</b>", section_style))
    elements.append(Spacer(1, 0.2 * inch))

    remarks_box = Table([[" "]], colWidths=[450], rowHeights=[80])
    remarks_box.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(remarks_box)
    elements.append(Spacer(1, 0.5 * inch))

    # -----------------------------
    # SIGNATURE
    # -----------------------------
    elements.append(Paragraph("______________________________", styles["Normal"]))
    elements.append(Paragraph("Radiologist Signature", styles["Normal"]))

    doc.build(elements)

    print(f"✅ PDF Generated: {filename}")