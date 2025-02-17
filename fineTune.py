import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)

model_id = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("squad_v2", split="train[:1%]")
val_dataset = load_dataset("squad_v2", split="validation[:1%]")


def format_example(example):
    question = example["question"]
    context = example["context"]

    # Some questions in SQuAD v2 are unanswerable, so we handle them.
    answer = (
        example["answers"]["text"][0] if example["answers"]["text"] else "No answer."
    )

    return {"text": f"Question: {question}\nContext: {context}\nAnswer: {answer}"}


def tokenize(example):
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=512
    )


train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.remove_columns(
    ["text", "question", "context", "answers", "id", "title"]
)
val_dataset = val_dataset.remove_columns(
    ["text", "question", "context", "answers", "id", "title"]
)
train_dataset.set_format("torch")
val_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./llama-qa-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    save_strategy="epoch",
    evaluation_strategy="epoch",
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
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("no runtime err")
