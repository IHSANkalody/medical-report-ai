import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Model
# -------------------------------------------------
model_name = "dmis-lab/biobert-base-cased-v1.1"

labels = ["O", "B-ORGAN", "I-ORGAN", "B-CONDITION", "I-CONDITION"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------------------------------------
# Dataset (Mini Example)
# -------------------------------------------------
data = {
    "tokens": [
        ["Liver", "shows", "mild", "fatty", "changes"],
        ["No", "free", "fluid", "in", "abdomen"],
        ["Gall", "bladder", "normal"],
        ["Both", "kidneys", "normal"]
    ],
    "ner_tags": [
        [1, 0, 3, 4, 4],
        [0, 3, 4, 0, 0],
        [1, 2, 0],
        [0, 1, 0]
    ]
}

dataset = Dataset.from_dict(data)

# -------------------------------------------------
# Tokenization + Label Alignment
# -------------------------------------------------
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True
    )

    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(example["ner_tags"][word_idx])
        else:
            label_ids.append(example["ner_tags"][word_idx])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels)

# -------------------------------------------------
# Model Initialization
# -------------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
).to(device)

# -------------------------------------------------
# Data Collator (IMPORTANT FIX)
# -------------------------------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)

# -------------------------------------------------
# Training Arguments
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=5,           # reduce epochs
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",
    report_to="none",             # disable reporting
    disable_tqdm=False,           # keep progress bar
    dataloader_num_workers=0,     # IMPORTANT for Windows
    learning_rate=5e-5
)


# -------------------------------------------------
# Trainer
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -------------------------------------------------
# Train
# -------------------------------------------------
trainer.train()
model.save_pretrained("./fine_tuned_biobert")
tokenizer.save_pretrained("./fine_tuned_biobert")

print("\n--- Testing After Fine-Tuning ---\n")

model.eval()

test_text = "Liver shows mild fatty changes"

inputs = tokenizer(test_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for token, pred in zip(tokens, predictions[0]):
    print(f"{token:15} → {id2label[pred.item()]}")
