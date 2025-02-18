import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# model_id = "meta-llama/Llama-3.2-1B"
model_id = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map=0
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("hellaswag", split="train[:100]", trust_remote_code=True)
eval_dataset = load_dataset(
    "hellaswag", split="validation[:100]", trust_remote_code=True
)


def preprocess_function(example):
    formatted_prompt = (
        f"Context: {example['ctx']}\n"
        "Choices:\n"
        f"A) {example['endings'][0]}\n"
        f"B) {example['endings'][1]}\n"
        f"C) {example['endings'][2]}\n"
        f"D) {example['endings'][3]}\n"
        f"Answer: "
    )

    # Map label index (0-3) to corresponding letter
    correct_choice = ["A", "B", "C", "D"][int(example["label"])]

    tokenized = tokenizer(
        formatted_prompt + correct_choice,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Remove extra dimension to avoid nested lists
    return {
        "input_ids": tokenized["input_ids"].squeeze(0).tolist(),
        "attention_mask": tokenized["attention_mask"].squeeze(0).tolist(),
        "labels": tokenized["input_ids"].squeeze(0).tolist(),
    }


train_dataset = train_dataset.map(preprocess_function, batched=False)
train_dataset = train_dataset.select_columns(["input_ids", "attention_mask", "labels"])
eval_dataset = eval_dataset.map(preprocess_function, batched=False)
eval_dataset = eval_dataset.select_columns(["input_ids", "attention_mask", "labels"])

# print("tokenized dataset sample", train_dataset[0])

# Data collator for efficient padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
# dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
# batch = next(iter(dataloader))
#
# print("processed dataset sample", batch)
#
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Adjust for GPU memory
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,  # Enable mixed precision for speedup
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
