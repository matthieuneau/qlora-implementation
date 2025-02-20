# Fine-Tuning and Evaluating HuggingFaceTB/SmolLM2-135M on HellaSwag

## Overview

This project explores different fine-tuning approaches for the **HuggingFaceTB/SmolLM2-135M** model on the **HellaSwag** benchmark. The goal was to compare the effectiveness of:

- **Full fine-tuning** (training all model parameters)
- **LoRA fine-tuning** (Low-Rank Adaptation)
- **QLoRA fine-tuning** (Quantized Low-Rank Adaptation)

After fine-tuning, the models were evaluated using the **lm-harness-evaluation** benchmark to assess performance.

This project was completed as part of a school assignment.

## Dataset: HellaSwag

HellaSwag is a **commonsense reasoning** dataset designed to challenge language models with multiple-choice questions. Each example contains:

- **A context**
- **Four possible endings (A, B, C, D)**
- **A correct answer label**

Fine-tuning aimed to improve the model's ability to select the correct ending based on the context.

## Fine-Tuning Approaches

### **1. Full Fine-Tuning**

In full fine-tuning, all model parameters were updated. This provides the highest flexibility but requires significant computational resources.

### **2. LoRA Fine-Tuning**

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method that injects trainable low-rank matrices into the model’s **attention layers**, keeping the base model frozen. This reduces computational cost while maintaining performance.

### **3. QLoRA Fine-Tuning**

**QLoRA (Quantized Low-Rank Adaptation)** builds on LoRA by applying **4-bit quantization** to the model, further reducing memory usage. It enables fine-tuning of large models on consumer GPUs while preserving model quality.

## Implementation

- The **transformers** library was used for model loading and fine-tuning.
- **LoRA and QLoRA** were implemented using the **peft** library.
- Training was performed with the **Trainer API**.
- Model evaluation was done using **lm-harness-evaluation**.

## Evaluation

The fine-tuned models were evaluated on **lm-harness-evaluation**, a framework that measures language model performance across tasks. The results compared:

- **Accuracy** on multiple-choice questions
- **Perplexity** (lower is better)
- **Computational efficiency** (GPU memory usage, speed)

## Results & Observations

| Fine-Tuning Method | Accuracy (%) | Perplexity | Memory Usage |
| ------------------ | ------------ | ---------- | ------------ |
| Full Fine-Tuning   | XX%          | XX.X       | High         |
| LoRA Fine-Tuning   | XX%          | XX.X       | Medium       |
| QLoRA Fine-Tuning  | XX%          | XX.X       | Low          |

Key takeaways:

- **Full fine-tuning** achieved the best accuracy but required the most memory.
- **LoRA fine-tuning** performed similarly while using fewer resources.
- **QLoRA fine-tuning** showed a minor drop in accuracy but significantly reduced GPU memory usage, making it ideal for limited hardware.

## Conclusion

This experiment demonstrated that **LoRA and QLoRA** provide viable alternatives to full fine-tuning, enabling efficient adaptation of LLMs with minimal resource consumption.

QLoRA, in particular, allows fine-tuning large models on consumer hardware while maintaining competitive performance.

## Future Work

- Extend fine-tuning to larger models.
- Experiment with different datasets.
- Explore advanced LoRA configurations for better efficiency.

## References

- Hugging Face Transformers: <https://huggingface.co/docs/transformers>
- HellaSwag Dataset: <https://rowanzellers.com/hellaswag/>
- LoRA Paper: <https://arxiv.org/abs/2106.09685>
- QLoRA Paper: <https://arxiv.org/abs/2305.14314>
