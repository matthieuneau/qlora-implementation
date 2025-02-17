import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)

model_id = "meta-llama/Llama-3.2-1B"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)


class CustomLlamaQA(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.qa_outputs = nn.Linear(base_model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = self.qa_outputs(outputs.last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


model = CustomLlamaQA(base_model)


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("squad_v2", split="train[:1%]")
val_dataset = load_dataset("squad_v2", split="validation[:1%]")


def tokenize(example):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True,
    )

    try:
        answer = example["answers"]["text"][0] if example["answers"]["text"] else None
        answer_start = (
            example["answers"]["answer_start"][0]
            if example["answers"]["answer_start"]
            else None
        )

        # Initialize labels as -100 (ignored in loss calculation)
        tokenized["start_positions"] = -100
        tokenized["end_positions"] = -100

        if answer:
            char_to_token_map = tokenized["offset_mapping"]
            for idx, (start, end) in enumerate(char_to_token_map):
                if start <= answer_start < end:
                    tokenized["start_positions"] = idx
                if start < answer_start + len(answer) <= end:
                    tokenized["end_positions"] = idx

        tokenized.pop("offset_mapping")  # Not needed for training

    except Exception as e:
        print("exception raised", e)
        print("problematic example", example["id"])

    return tokenized


# TODO: Fix the tokenize function so that it works with batched=False
train_dataset = train_dataset.map(tokenize, batched=False)
val_dataset = val_dataset.map(tokenize, batched=False)
train_dataset = train_dataset.remove_columns(
    ["question", "context", "answers", "id", "title"]
)
val_dataset = val_dataset.remove_columns(
    ["context", "question", "answers", "id", "title"]
)
train_dataset.set_format("torch")
val_dataset.set_format("torch")

print("ok")

training_args = TrainingArguments(
    output_dir="./llama-qa-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    fp16=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
