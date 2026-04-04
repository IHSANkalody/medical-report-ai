import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "dmis-lab/biobert-base-cased-v1.1"

# Define BIO label set for ultrasound domain
labels = ["O", "B-ORGAN", "I-ORGAN", "B-CONDITION", "I-CONDITION"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
).to(device)

text = "Liver shows mild fatty changes"

inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model(**inputs)
logits = outputs.logits

predictions = torch.argmax(logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for token, pred in zip(tokens, predictions[0]):
    print(f"{token:15} → {id2label[pred.item()]}")
