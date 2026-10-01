from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import os

app = FastAPI(title="Harpertoken QA API", version="0.1.0")


class QAPredictRequest(BaseModel):
    question: str
    context: str


# Load model. The question-answering pipeline was removed in transformers 5,
# so inference goes through AutoModelForQuestionAnswering directly.
model_path = "./results/checkpoint-500"
source = model_path if os.path.exists(model_path) else "harpertoken/quiz"
tokenizer = AutoTokenizer.from_pretrained(source)
model = AutoModelForQuestionAnswering.from_pretrained(source)
model.eval()


def answer_span(question, context, max_answer_len=30):
    """Return the best contiguous span and its score.

    The start and end logits are searched independently by the usual
    max-over-span approach, which can select end < start. Constraining the
    search to start <= end and to a plausible span length avoids decoding
    empty or reversed answers.
    """
    inputs = tokenizer(
        question, context, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.inference_mode():
        out = model(**inputs)

    seq_len = inputs["input_ids"].shape[1]
    start_logits = out.start_logits[0]
    end_logits = out.end_logits[0]

    mask = inputs["attention_mask"][0].bool()
    start_logits = start_logits.masked_fill(~mask, float("-inf"))
    end_logits = end_logits.masked_fill(~mask, float("-inf"))

    # Best (start, end) pair with start <= end and a bounded span.
    span_limit = min(max_answer_len, seq_len - 1)
    valid_ends = end_logits.unsqueeze(0) + start_logits.unsqueeze(1)
    upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    valid_ends = valid_ends.masked_fill(upper, float("-inf"))
    # Exclude spans longer than max_answer_len.
    too_long = torch.arange(seq_len).unsqueeze(1) + span_limit < torch.arange(
        seq_len
    ).unsqueeze(0)
    valid_ends = valid_ends.masked_fill(too_long, float("-inf"))

    best = int(valid_ends.argmax())
    start, end = best // seq_len, best % seq_len
    score = float((start_logits[start] + end_logits[end]) / 2)
    text = tokenizer.decode(
        inputs["input_ids"][0][start : end + 1], skip_special_tokens=True
    )
    return text.strip(), score


@app.post("/predict")
def predict(request: QAPredictRequest):
    text, score = answer_span(request.question, request.context)
    return {"answer": text, "score": score}


@app.get("/")
def root():
    return {"message": "Harpertoken QA API", "endpoints": ["/predict"]}
