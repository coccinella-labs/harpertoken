# Contributing

1. Fork the repo.
2. Create a feature branch.
3. Make changes and test.
4. Submit a pull request with conventional commit messages.

For issues, open a [GitHub issue](https://github.com/coccinella-labs/harpertoken/issues).

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

## Testing

Run tests locally:
```sh
pip install pytest
pytest tests/
```

Tests include data preprocessing validation.

## CI/CD

This project uses GitHub Actions for continuous integration, model training, and Docker for containerization.

- **Linting**: Flake8, Black, MyPy on every push/PR.
- **Testing**: Pytest unit tests, syntax checks, import tests, and post-training validation.
- **Training**: Fine-tunes the model on a SQuAD subset with configurable epochs/batch/LR.
- **Hugging Face Upload**: Pushes fine-tuned model and tokenizer to HF Hub with model card (requires `HF_TOKEN` secret).
- **Docker Build**: Builds and pushes image to Docker Hub and GHCR on main branch.

Workflows: `.github/workflows/ci.yml` (CI/training), `.github/workflows/docs.yml` (MkDocs deployment to GitHub Pages on harpertoken branch)

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
