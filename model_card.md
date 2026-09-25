---
library_name: transformers
tags:
- question-answering
- distilbert
- squad
- fine-tuned
datasets:
- squad
---

# Model Card for harpertoken/harpertokenConvAI-finetuned

This model is a fine-tuned version of harpertoken/harpertokenConvAI, a DistilBERT-based question answering model, trained on a subset of the SQuAD dataset.

## Model Details

### Model Description

This is a fine-tuned question answering model based on DistilBERT, optimized for extractive QA tasks. It has been trained on a small subset of the SQuAD dataset to demonstrate fine-tuning capabilities in a CI environment.

- **Developed by:** Coccinella Labs
- **Model type:** DistilBERT for Question Answering
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from model:** harpertoken/harpertokenConvAI

### Model Sources

- **Repository:** https://github.com/coccinella-labs/harpertoken

## Uses

### Direct Use

This model can be used directly for question answering on passages similar to SQuAD. Provide a question and context, and it will predict the answer span.

### Downstream Use

Can be further fine-tuned on domain-specific data for improved performance.

### Out-of-Scope Use

Not suitable for non-English text, generative tasks, or domains outside of factual QA.

## Bias, Risks, and Limitations

Trained on a limited SQuAD subset, may exhibit biases from the dataset. Performance may degrade on out-of-domain questions.

### Recommendations

Evaluate on your specific data and consider additional fine-tuning for production use.

## How to Get Started with the Model

```python
from transformers import pipeline

qa = pipeline("question-answering", model="harpertoken/harpertokenConvAI-finetuned")
result = qa(question="What is the capital of France?", context="France is a country in Europe. Paris is the capital.")
print(result)
```

## Training Details

### Training Data

Subset of SQuAD 1.1 dataset (approximately 1000 examples).

### Training Procedure

#### Training Hyperparameters

- **Training regime:** fp32
- **Epochs:** 1
- **Batch size:** 1
- **Learning rate:** 2e-5

#### Speeds, Sizes, Times

Trained in CI environment, minimal time due to small dataset.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

SQuAD validation set subset.

#### Metrics

F1 score, Exact Match.

### Results

Basic evaluation on sample questions.

## Environmental Impact

Minimal impact due to small-scale training in CI.

- **Hardware Type:** GitHub Actions runners
- **Carbon Emitted:** Negligible

## Technical Specifications

### Model Architecture and Objective

DistilBERT encoder with QA head for span prediction.

### Compute Infrastructure

GitHub Actions Ubuntu runners.

## Citation

If you use this model, please cite the original DistilBERT and SQuAD papers.

## Model Card Contact

Coccinella Labs
