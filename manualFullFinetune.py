import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.optimization import get_scheduler

import wandb

wandb.init(project="qlora")

config_file = "config.yaml"
try:
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    config = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = config.get("model", "HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map=device
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

train_split = config.get("train_split", "train[:100]")
eval_split = config.get("eval_split", "validation[:100]")
train_dataset = load_dataset("hellaswag", split=train_split, trust_remote_code=True)
eval_dataset = load_dataset("hellaswag", split=eval_split, trust_remote_code=True)


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

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.get("batch_size", 1),
    shuffle=True,
    collate_fn=data_collator,
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=config.get("batch_size", 1),
    shuffle=False,
    collate_fn=data_collator,
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 2e-5))
num_train_epochs = config.get("num_train_epochs", 5)
num_training_steps = len(train_loader) * num_train_epochs

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)


model.train()
global_step = 0
for epoch in range(num_train_epochs):
    running_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}


trainer.train()
