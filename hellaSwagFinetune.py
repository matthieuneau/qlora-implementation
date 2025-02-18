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
    model_id, torch_dtype=torch.bfloat16, device_map=0
)
