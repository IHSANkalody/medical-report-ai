from report_generator import extract_findings
from pdf_generator import generate_pdf

# ----------------------------
# MAIN ENTRY POINT
# ----------------------------
if __name__ == "__main__":

    text = input("Enter ultrasound findings: ")

    findings = extract_findings(text)

    print("\nExtracted Findings:\n", findings)

    generate_pdf(findings)