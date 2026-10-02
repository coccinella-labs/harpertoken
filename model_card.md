---
license: mit
language: en
library_name: transformers
pipeline_tag: feature-extraction
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

<!-- SOURCE OF TRUTH: this file. Do not edit harpertoken/clue's card on the Hub
     directly; the sync-card CI job overwrites it from here on every push to main. -->

# clue

A short continued-fine-tuning run of [`quiz`](https://huggingface.co/harpertoken/quiz), itself a DistilBERT encoder adapted for extractive question answering on SQuAD. The architecture and tokenizer are identical; only the weights differ. Where `quiz` reflects a full training pass, this checkpoint reflects roughly a thousand SQuAD examples seen once, which makes it a useful small-scale reference point and a poor substitute for a properly trained model.

Training used a learning rate of 2e-5 at batch size one for a single epoch, in float32. The published weights are `model.safetensors`. The `config.json` previously carried a key `tie_weights_`, which no version of Transformers reads; it has been removed, and nothing else in the config was altered.

## Usage

Transformers 5 removed the `question-answering` pipeline, so load the model directly:

```python
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

tok = AutoTokenizer.from_pretrained("harpertoken/clue")
model = AutoModelForQuestionAnswering.from_pretrained("harpertoken/clue")

question = "Who wrote Hamlet?"
context = "Hamlet is a tragedy written by William Shakespeare around 1600."
inputs = tok(question, context, return_tensors="pt", truncation=True, max_length=512)

with torch.inference_mode():
    out = model(**inputs)
start, end = int(out.start_logits.argmax()), int(out.end_logits.argmax())
print(tok.decode(inputs.input_ids[0][start : end + 1]))
```

On the three questions used to check `quiz` (the capital of France, the author of Hamlet, and the year the Eiffel Tower was completed), this checkpoint returns `paris`, `william shakespeare` and `1889`, the same answers. A thousand examples has not visibly degraded it, which is itself a reason to doubt that the fine-tuning taught much.

## Limitations

A thousand examples is a demonstration of the fine-tuning mechanics rather than a trained model, and the documentation this replaced claimed SQuAD exact-match and F1 figures that were never produced by an evaluation. Treat this as a low-fidelity copy of `quiz`. It is English-only, inherits the same uncased tokenisation and SQuAD domain bias described in the `quiz` card, and shares its 512-token limit. Compare the two directly before assuming the fine-tuning helped.

## Attribution

DistilBERT follows Sanh et al. (2019); SQuAD follows Rajpurkar et al. (2016).
