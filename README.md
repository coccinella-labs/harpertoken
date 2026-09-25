<p align="center">
  <img src="https://raw.githubusercontent.com/basebin/harpertoken/main/.github/assets/thumbnail.png" alt="harpertoken" width="100%">
</p>

# Harpertoken ConvAI Fine-tuning

[![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project provides scripts to fine-tune the `harpertokenConvAI` model (a DistilBERT-based question answering model) on a subset of the SQuAD dataset, optimized for Mac M1 with 8GB RAM using MPS acceleration.

The full API and scripts are organized for easy extension and customization.

## Installation

1. **Clone or set up the project** (assuming local setup):
   ```sh
   # Project is already in /Users/niladri/Desktop/model
   cd /Users/niladri/Desktop/model
   ```

2. **Create Virtual Environment**:
   ```sh
   python3 -m venv venv
   ```

3. **Activate Virtual Environment**:
   ```sh
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   # Optional: Install code quality tools
   pip install black flake8 mypy
   ```

## Usage

The builder script (`run.sh`) orchestrates the fine-tuning process with configurable options.

### Basic Usage
```sh
./run.sh
```

### Interactive Configuration
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

### API Serving
After training, serve the model via API:
```sh
pip install fastapi uvicorn
python scripts/api.py  # Or uvicorn scripts.api:app --reload
```

API Endpoints:
- `GET /`: API info
- `POST /predict`: QA prediction (json: {"question": "...", "context": "..."})

### Configuration File

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
- Optionally upload to `harpertoken/harpertokenConvAI-finetuned`.

### Uploading to Hugging Face

Set `HF_TOKEN` env var for uploads:
```sh
export HF_TOKEN=your_token
./run.sh --upload
```

Get token from [Hugging Face settings](https://huggingface.co/settings/tokens).

### Manual Usage

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

### Request & Response Types

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

## Project Structure

- `scripts/`: Python scripts for data prep, training, and evaluation.
- `data/`: Placeholder for custom datasets.
- `models/`: Placeholder for saved models.
- `results/`: Training outputs and checkpoints.
- `__version__.py`: Version information.
- `requirements.txt`: Dependencies.
- `pyproject.toml`: Project configuration and tool settings.
- `config.yaml`: Default configuration for builds.
- `Dockerfile`: Docker container configuration.
- `run.sh`: Orchestration script.
- `scripts/`: Python scripts for fine-tuning.
- `tests/`: Unit tests.
- `.github/`: GitHub Actions workflows.
- `venv/`: Virtual environment.

## CI/CD

This project uses GitHub Actions for continuous integration and Docker for containerization.

- **Linting**: Flake8, Black, MyPy on every push/PR.
- **Testing**: Pytest unit tests, syntax checks, and import tests.
- **Docker Build**: Builds and pushes image to Docker Hub and GHCR on main branch.

Workflow: `.github/workflows/ci.yml`

## Docker

Build and run the project in a container.

### Build Locally
```sh
docker build -t harpertoken-convai-finetune .
```

### Run
```sh
docker run --rm harpertoken-convai-finetune
```

### Pull from Registry
- Docker Hub: `docker pull <username>/harpertoken-convai-finetune`
- GHCR: `docker pull ghcr.io/<username>/<repo>/harpertoken-convai-finetune`

Dockerfile: `Dockerfile`

## Testing

Run tests locally:
```sh
pip install pytest
pytest tests/
```

Tests include data preprocessing validation.

## Requirements

- Python >= 3.14
- PyTorch with MPS support (Mac M1)
- 8GB RAM minimum
- Supported runtimes: macOS with Apple Silicon

## Git Hooks and Conventional Commits

This project uses git hooks for code quality and conventional commit standards.

### Setup

1. **Enable Pre-commit Hook** (code quality):
   ```sh
   cp scripts/pre-commit .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   ```

2. **Enable Commit Hook** (conventional commits):
   ```sh
   cp scripts/commit-msg .git/hooks/commit-msg
   chmod +x .git/hooks/commit-msg
   ```

3. **Rewrite Existing Messages** (if needed):
   ```sh
   ./scripts/rewrite_msg.sh <commit-hash>  # For single commit
   ./scripts/rewrite_msg.sh <start>..<end>  # For range
   git push --force  # After rewriting
   ```

### Commit Message Format

- Start with type: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`, `perf:`, `ci:`, `build:`, `revert:`
- Description in lowercase
- First line ≤40 characters

Example: `feat: add conventional commit hook`

## Contributing

1. Fork the repo.
2. Create a feature branch.
3. Make changes and test.
4. Submit a pull request with conventional commit messages.

For issues, open a [GitHub issue](https://github.com/coccinella-labs/path/issues) (adapt to your repo).

## Frequently Asked Questions

**Q: How to increase training data?**
A: Edit `select(range(1000))` in `data_prep.py` to a larger range.

**Q: Can this run on other hardware?**
A: Yes, modify device to 'cuda' for GPU or 'cpu' for CPU-only.

**Q: What if I want to deploy the model?**
A: Use Hugging Face's model upload after training.

## Semantic Versioning

This project follows SemVer for releases. Backwards-incompatible changes will be in major versions.

We appreciate feedback; open an issue for suggestions.
