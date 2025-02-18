from datasets import load_dataset
from evaluate import evaluator

task_evaluator = evaluator("question-answering")

data = load_dataset("squad", split="validation[:1000]")
eval_results = task_evaluator.compute(
    model_or_pipeline="meta-llama/Llama-3.2-1B",
    data=data,
    metric="squad",
    strategy="bootstrap",
    n_resamples=30,
)

print(eval_results)
