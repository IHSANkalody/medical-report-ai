import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./fine_tuned_biobert"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)

labels = ["O", "B-ORGAN", "I-ORGAN", "B-CONDITION", "I-CONDITION"]

# ------------------------------------------------
# Improved Extraction Logic (AI + Rules)
# ------------------------------------------------
def extract_findings(text):

    findings = {
        "Liver": "Normal",
        "Gall Bladder": "Normal",
        "Kidneys": "Normal",
        "Free Fluid": "Absent"
    }

    lower_text = text.lower()

    # Liver conditions
    if any(word in lower_text for word in ["fatty", "hepatomegaly", "enlarged", "echogenicity"]):
        findings["Liver"] = "Abnormal"

    # Gall bladder conditions
    if any(word in lower_text for word in ["calculi", "stone", "thickened"]):
        findings["Gall Bladder"] = "Abnormal"

    # Kidney conditions
    if any(word in lower_text for word in ["hydronephrosis", "dilated", "cyst"]):
        findings["Kidneys"] = "Abnormal"

    # Free fluid handling
    if "free fluid" in lower_text:
        if any(neg in lower_text for neg in ["no free fluid", "absent free fluid", "no significant free fluid"]):
            findings["Free Fluid"] = "Absent"
        else:
            findings["Free Fluid"] = "Present"

    return findings


# ------------------------------------------------
# Report Generator
# ------------------------------------------------
def generate_report(findings):

    report = f"""
----------------------------------------
        ULTRASOUND ABDOMEN REPORT
----------------------------------------

Liver        : {findings['Liver']}
Gall Bladder : {findings['Gall Bladder']}
Kidneys      : {findings['Kidneys']}
Free Fluid   : {findings['Free Fluid']}

----------------------------------------
"""

    return report


# ------------------------------------------------
# Demo Run
# ------------------------------------------------
if __name__ == "__main__":

    text = input("Enter ultrasound findings: ")

    findings = extract_findings(text)

    report = generate_report(findings)

    print(report)