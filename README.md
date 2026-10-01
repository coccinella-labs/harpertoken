<p align="center">
  <img src="https://raw.githubusercontent.com/coccinella-labs/harpertoken/main/.github/assets/thumbnail.png" alt="harpertoken" width="100%">
</p>

# Harpertoken ConvAI fine-tuning

Fine-tuning scripts for [`quiz`](https://huggingface.co/harpertoken/quiz), the extractive
question answering model built on DistilBERT and trained on SQuAD. The target machine is
an Apple Silicon Mac with 8 GB of RAM.

The repository holds a data preparation step, a training loop, an evaluation script, and a
FastAPI service that wraps the trained checkpoint. A shell script chains the first three
together behind an interactive prompt.

## Requirements

Python 3.10 or newer, PyTorch, and the dependencies listed in `requirements.txt`. MPS
acceleration is used on Apple Silicon, CUDA on NVIDIA, and CPU otherwise. Expect to need
roughly 8 GB of RAM at the default settings.

## Installation

```sh
git clone https://github.com/coccinella-labs/harpertoken.git
cd harpertoken
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`run.sh` sources `venv/bin/activate` on every run, so the environment has to exist at that
path before you use it.

## Usage

### The interactive script

`./run.sh` asks for the dataset, the task, whether to tune hyperparameters, and, when
tuning is off, the epoch count, batch size, learning rate, and whether to upload to the
Hub. It accepts no command line flags. Defaults are 1 epoch, batch size 2, learning rate
2e-5, no tuning and no upload, and they come from `config.yaml` through
`scripts/load_config.py`.

Because there are no flags, a non interactive run is expressed through the environment:

```sh
FT_EPOCHS=1 FT_BATCH_SIZE=1 FT_UPLOAD=false ./run.sh
```

### Training

`scripts/train.py` fine tunes `harpertoken/quiz` on 1,000 SQuAD training examples and 200
validation examples. The subset sizes are what keep the run inside 8 GB, and both are set
in `scripts/data_prep.py:10`. Results, including the checkpoint used by the other
scripts, are written to `results/`.

```sh
python scripts/train.py
```

### Evaluation

`scripts/evaluate.py` reads the checkpoint from `results/`, answers two questions, and
reports exact match and F1 against the expected answers:

```sh
python scripts/evaluate.py
```

Both scores come out at 1.0000. That is a two question smoke test that confirms the
pipeline is wired up, and it is not a benchmark. Score against the SQuAD v1.1 dev set if
you need a real number.

### Serving

`scripts/api.py` defines a FastAPI app, so it needs uvicorn to run. Executing the file
directly does nothing.

```sh
uvicorn scripts.api:app --reload
```

| Endpoint | Purpose |
|---|---|
| `GET /` | Service name and endpoint list |
| `POST /predict` | Answer extraction from `{"question": ..., "context": ...}` |

The service prefers `results/checkpoint-500` when that directory exists and otherwise
falls back to `harpertoken/quiz`, so it responds before you have trained anything.

Inference goes through `AutoModelForQuestionAnswering` because transformers 5 removed the
`question-answering` pipeline task. Pinning below version 5 would only postpone the same
break.

### Uploading

Set `HF_TOKEN` and either answer yes to the prompt or set the variable:

```sh
export HF_TOKEN=your_token          # https://huggingface.co/settings/tokens
FT_UPLOAD=true ./run.sh
```

This pushes the checkpoint to `harpertoken/clue` and overwrites that repository's weights
and model card. Change `scripts/train.py:101` if the output needs to go elsewhere.

## Layout

| Path | Contents |
|---|---|
| `scripts/data_prep.py` | SQuAD loading and tokenisation, and the subset sizes |
| `scripts/train.py` | Fine tuning loop and Hub upload |
| `scripts/evaluate.py` | Exact match and F1 against sample questions |
| `scripts/api.py` | FastAPI service, and the span decoder it shares with evaluation |
| `scripts/load_config.py` | Reads `config.yaml` into environment variables |
| `tests/` | Pytest suite |
| `run.sh` | Interactive orchestration of train then evaluate |
| `config.yaml` | Default settings |
| `Dockerfile` | Container image |

`results/` is created by training and is git ignored.

## Testing and CI

```sh
pip install pytest pytest-cov
pytest tests/
```

`.github/workflows/ci.yml` runs pre commit on every push, then tests on Python 3.10 and
3.11. The training step in CI runs with `FT_UPLOAD=true`, so every push to `main` retrains
and reuploads `harpertoken/clue`. The Hub model is therefore not a stable artifact between
runs.

The release job builds a Docker image only when the version in `__version__.py` has no
matching git tag, and creates the release tag itself.

## Docker

```sh
docker build -t harpertoken-convai-finetune .
docker run --rm harpertoken-convai-finetune
```

Images are published to Docker Hub and GitHub Container Registry as
`harpertoken-convai-finetune`, tagged with the release version and with `latest`.

## Troubleshooting

**Out of memory.** Lower `per_device_train_batch_size` in `scripts/train.py`, or leave it
at 1.

**MPS errors.** Confirm the installed PyTorch build includes MPS support. CPU execution
works but is slower, and `scripts/evaluate.py` selects the device on its own.

**`FileNotFoundError` for `results/`.** Evaluation reads from `results/` and takes no
arguments, so train first or edit the `model_path` default at the top of
`scripts/evaluate.py`.

**`ModuleNotFoundError`.** The virtual environment is not active, or
`pip install -r requirements.txt` has not been run inside it.

## Contributing

Fork, branch, test, and open a pull request. Commits follow
[Conventional Commits](https://www.conventionalcommits.org/), enforced locally once the
hooks are installed:

```sh
cp scripts/pre-commit .git/hooks/pre-commit
cp scripts/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/pre-commit .git/hooks/commit-msg
```

## License

MIT. See [LICENSE](LICENSE).
