import torch
from datasets import Dataset
from transformers import AutoTokenizer

model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

labels = ["O", "B-ORGAN", "I-ORGAN", "B-CONDITION", "I-CONDITION"]
label2id = {l: i for i, l in enumerate(labels)}

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

print(tokenized_dataset[0])
