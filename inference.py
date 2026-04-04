import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./fine_tuned_biobert"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)

labels = ["O", "B-ORGAN", "I-ORGAN", "B-CONDITION", "I-CONDITION"]

text = "Liver shows mild fatty changes. No free fluid in abdomen."

inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for token, pred in zip(tokens, predictions[0]):
    print(token, "→", labels[pred.item()])
