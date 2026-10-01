from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI(title="Harpertoken QA API", version="0.1.0")


class QAPredictRequest(BaseModel):
    question: str
    context: str


# Load model
model_path = "./results/checkpoint-500"
if os.path.exists(model_path):
    qa_pipeline = pipeline("question-answering", model=model_path, device=-1)
else:
    qa_pipeline = pipeline("question-answering", model="harpertoken/quiz", device=-1)


@app.post("/predict")
def predict(request: QAPredictRequest):
    result = qa_pipeline(question=request.question, context=request.context)
    return {"answer": result["answer"], "score": result["score"]}


@app.get("/")
def root():
    return {"message": "Harpertoken QA API", "endpoints": ["/predict"]}
