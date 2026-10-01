# Usage

The builder script (`run.sh`) orchestrates the fine-tuning process with configurable options.

## Basic Usage
```sh
./run.sh
```

## Interactive Configuration
When you run `./run.sh`, it will prompt you for configuration options interactively:

- **Dataset**: Choose the dataset (default: squad)
- **Task**: Task type (default: qa)
- **Tune**: Enable hyperparameter tuning with Optuna (y/n, default: n)
- If tuning is disabled:
  - **Epochs**: Number of training epochs (default: 1)
  - **Batch**: Batch size (default: 2)
  - **LR**: Learning rate (default: 2e-5)
- **Upload**: Upload to Hugging Face after training (y/n, default: n)

The script will then proceed with training and evaluation based on your inputs.

## Configuration File

Use `config.yaml` for persistent settings:
```yaml
dataset: squad
epochs: 2
batch_size: 4
learning_rate: 0.00005
upload: true
```

The CLI loads from `config.yaml` and uses as prompt defaults.

This will:
- Load and preprocess the dataset.
- Fine-tune the model with specified params.
- Evaluate on sample questions.
- Optionally upload to `harpertoken/clue`.

## Uploading to Hugging Face

Set `HF_TOKEN` env var for uploads:
```sh
export HF_TOKEN=your_token
./run.sh --upload
```

Get token from [Hugging Face settings](https://huggingface.co/settings/tokens).

## Manual Usage

If you prefer to run scripts individually:

1. **Train the Model**:
    ```sh
    python scripts/train.py
    ```
    Loads the model, preprocesses a small SQuAD subset, and fine-tunes for 1 epoch.

2. **Evaluate the Model**:
    ```sh
    python scripts/evaluate.py
    ```
    Loads the fine-tuned model from `results/` and answers sample questions.

## Request & Response Types

The scripts use standard Hugging Face transformers types. For custom datasets, modify `scripts/data_prep.py` to return tokenized datasets with required fields (`input_ids`, `attention_mask`, `start_positions`, `end_positions`).

## Handling Errors

If training fails due to memory issues, reduce `per_device_train_batch_size` in `scripts/train.py`. For MPS errors, ensure PyTorch is installed with MPS support.

Common errors:
- `CUDA out of memory`: Reduce batch size.
- `ModuleNotFoundError`: Ensure venv is activated and dependencies installed.

## Advanced Usage

### Customizing Training

Edit `scripts/train.py` to adjust:
- `num_train_epochs`: Increase for better performance (monitor RAM).
- `learning_rate`: Tune for convergence.
- Dataset: Modify `data_prep.py` to use custom QA datasets.

### Logging

Training logs are printed to console. For more verbose logging, set `logging_steps` lower in `TrainingArguments`.

### Accessing Raw Outputs

The evaluation script prints answers with confidence scores. To access raw model outputs, modify `scripts/evaluate.py` to return full predictions.

### Custom Datasets

To use a custom dataset:
1. Update `load_and_preprocess_data` in `scripts/data_prep.py`.
2. Ensure the dataset has `question`, `context`, and `answers` columns.
