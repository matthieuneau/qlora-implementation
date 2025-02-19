import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
import wandb

wandb.init(project="qlora")

model_id = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # load_in_4bit=True,  # Use 4-bit quantization for efficient training
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("hellaswag", split="train[:1000]", trust_remote_code=True)
eval_dataset = load_dataset(
    "hellaswag", split="validation[:200]", trust_remote_code=True
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

    correct_choice = ["A", "B", "C", "D"][int(example["label"])]

    tokenized = tokenizer(
        formatted_prompt + correct_choice,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": tokenized["input_ids"].squeeze(0).tolist(),
        "attention_mask": tokenized["attention_mask"].squeeze(0).tolist(),
        "labels": tokenized["input_ids"].squeeze(0).tolist(),
    }


train_dataset = train_dataset.map(preprocess_function, batched=False)
train_dataset = train_dataset.select_columns(["input_ids", "attention_mask", "labels"])
eval_dataset = eval_dataset.map(preprocess_function, batched=False)
eval_dataset = eval_dataset.select_columns(["input_ids", "attention_mask", "labels"])


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

training_args = TrainingArguments(
    output_dir="./results_lora",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # Higher LR for LoRA
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=1,  # only save last epoch
    eval_strategy="epoch",
    fp16=False,
    bf16=True,
    bf16_full_eval=True,
    num_train_epochs=3,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_lora_model")
tokenizer.save_pretrained("./merged_lora_model")
