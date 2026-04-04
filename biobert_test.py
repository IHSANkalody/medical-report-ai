from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Public, stable NER model (no authentication needed)
model_name = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0  # GPU
)

text = "Liver shows mild fatty changes. No free fluid in abdomen."

results = ner(text)

for r in results:
    print(r)
