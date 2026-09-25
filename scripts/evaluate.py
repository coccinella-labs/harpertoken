import torch
from transformers import (
    AutoTokenizer,
    DistilBertForQuestionAnswering,
    pipeline,
)  # noqa: E501

# Example questions with ground truth
QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Europe. Its capital is Paris.",
        "answers": ["Paris"],
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "context": (
            "William Shakespeare wrote many plays, including Romeo and Juliet."
        ),
        "answers": ["William Shakespeare", "Shakespeare"],
    },
]


def compute_f1(pred, true):
    pred_tokens = pred.lower().split()
    true_tokens = set(true.lower().split())
    common = set(pred_tokens) & true_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def main(model_path="./results"):
    # Load fine-tuned model and tokenizer from checkpoint
    model = DistilBertForQuestionAnswering.from_pretrained(
        model_path, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )

    # Create QA pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Compute metrics
    exact_matches = 0
    f1_scores = []

    # Answer questions
    for q in QUESTIONS:
        answer = qa_pipeline(question=q["question"], context=q["context"])
        pred = answer["answer"]
        true_answers = q["answers"]
        # Check exact match
        em = any(pred.strip().lower() == t.lower() for t in true_answers)
        exact_matches += int(em)
        # Compute F1 (take max over possible answers)
        f1 = max(compute_f1(pred, t) for t in true_answers)
        f1_scores.append(f1)
        print(f"Question: {q['question']}")
        print(f"Predicted: {pred}")
        print(f"Ground Truth: {true_answers}")
        print(f"Exact Match: {em}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confidence: {answer['score']:.4f}")
        print("-" * 50)

    # Overall metrics
    avg_em = exact_matches / len(QUESTIONS)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"Overall Exact Match: {avg_em:.4f}")
    print(f"Overall F1 Score: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
