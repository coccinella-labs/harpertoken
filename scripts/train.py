import os
import torch
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    DistilBertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from scripts.data_prep import load_and_preprocess_data

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "harpertoken/quiz"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name).to(device)

# Load and preprocess data
tokenized_squad = load_and_preprocess_data(tokenizer)
print("Train columns:", tokenized_squad["train"].column_names)
print("Validation columns:", tokenized_squad["validation"].column_names)

# Data collator
data_collator = DefaultDataCollator()

# Read config from env
tune = os.getenv("FT_TUNE", "false").lower() == "true"
if tune:
    import optuna

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # noqa: F841
        batch_size = trial.suggest_categorical(  # noqa: F841,E501
            "batch_size", [2, 4, 8]
        )

        epochs = trial.suggest_int("epochs", 1, 3)  # noqa: F841
        # Train with these params and return eval_loss
        # Simplified: just return a mock score
        return 1.0  # Replace with actual training

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best = study.best_params
    lr = best["lr"]
    batch_size = best["batch_size"]
    epochs = best["epochs"]
    print(f"Best params: {best}")
else:
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(os.getenv("FT_LR", 2e-5)),
        per_device_train_batch_size=int(os.getenv("FT_BATCH_SIZE", 2)),
        per_device_eval_batch_size=int(os.getenv("FT_BATCH_SIZE", 2)),
        num_train_epochs=int(os.getenv("FT_EPOCHS", 1)),
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=50,
        load_best_model_at_end=True,
        remove_unused_columns=False,  # Keep all columns
        dataloader_pin_memory=False,  # Disable for MPS
    )

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model
trainer.save_model("./results")
tokenizer.save_pretrained("./results")

# Push to Hugging Face if requested
upload = os.getenv("FT_UPLOAD", "false").lower() == "true"
if upload:
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        model_card = None
        if os.path.exists("model_card.md"):
            with open("model_card.md", "r") as f:
                model_card = f.read()
        model.push_to_hub("harpertoken/clue", token=hf_token)
        tokenizer.push_to_hub("harpertoken/clue", token=hf_token)  # noqa: E501
        if model_card:
            # push_to_hub generates its own card and, before Transformers 5,
            # silently absorbed a `model_card` argument through **deprecated_kwargs.
            # Transformers 5 rejects it, so upload ours separately. It has to be
            # after the pushes, or those overwrite the README.
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id="harpertoken/clue",
                repo_type="model",
            )
        print(
            "Model and tokenizer pushed to "
            "harpertoken/clue"  # noqa: E501
        )
    else:
        print("HF_TOKEN not set. Cannot upload to Hugging Face.")
else:
    print("Upload not requested. Skipping Hugging Face upload.")
