---
license: mit
language: en
library_name: transformers
pipeline_tag: question-answering
base_model: distilbert-base-uncased
datasets:
- squad
tags:
- distilbert
- extractive-qa
- question-answering
- squad
- fine-tuned
---

# clue

A short continued-fine-tuning run of [`quiz`](https://huggingface.co/harpertoken/quiz), itself a DistilBERT encoder adapted for extractive question answering on SQuAD. The architecture and tokenizer are identical; only the weights differ. Where `quiz` reflects a full training pass, this checkpoint reflects roughly a thousand SQuAD examples seen once, which makes it a useful small-scale reference point and a poor substitute for a properly trained model.

Training used a learning rate of 2e-5 at batch size one for a single epoch, in float32. The published weights are `model.safetensors`. The `config.json` previously carried a key `tie_weights_`, which no version of Transformers reads; it has been removed, and nothing else in the config was altered.

## Usage

```python
from transformers import pipeline

qa = pipeline("question-answering", model="harpertoken/clue")
answer = qa(
    question="What is the capital of France?",
    context="France is a country in Europe. Paris is its capital.",
)
print(answer["answer"], answer["score"])
```

## Limitations

A thousand examples is a demonstration of the fine-tuning mechanics rather than a trained model, and the documentation this replaced claimed SQuAD exact-match and F1 figures that were never produced by an evaluation. Treat this as a low-fidelity copy of `quiz`. It is English-only, inherits the same uncased tokenisation and SQuAD domain bias described in the `quiz` card, and shares its 512-token limit. Compare the two directly before assuming the fine-tuning helped.

## Attribution

DistilBERT follows Sanh et al. (2019); SQuAD follows Rajpurkar et al. (2016).
