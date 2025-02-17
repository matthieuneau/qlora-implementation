import torch
from transformers import pipeline, AutoConfig, AutoModel

model_id = "meta-llama/Llama-3.2-1B"

config = AutoConfig.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# print(config)
print(model)

# pipe = pipeline(
#     "text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto"
# )
#
# print(pipe("The key to success is"))
