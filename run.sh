#!/bin/bash

# Activate venv
source venv/bin/activate

# Install deps if needed
python3 -c "import yaml, transformers, torch" 2>/dev/null || pip install -r requirements.txt

# Load version and config. load_config.py must be eval'd rather than just run:
# environment variables set inside that Python process die with it, so a bare
# call leaves config.yaml silently inert.
VERSION=$(python3 -c "import re; print(re.search(r'(?<=^version = \")[^\"]+', open('pyproject.toml').read(), re.M).group(0))")
eval "$(python3 scripts/load_config.py --export)"

# Header
echo "Harpertoken CLI v$VERSION"

# Prompt function
prompt() { read -p "$1 [$2]: " input; echo "${input:-$2}"; }

# Config prompts
DATASET=$(prompt "Dataset" "${FT_DATASET:-squad}")
TASK=$(prompt "Task" "${FT_TASK:-qa}")
TUNE_CHOICE=$(prompt "Tune" "$([ "${FT_TUNE:-false}" = "true" ] && echo "y" || echo "n")")
TUNE=$([ "$TUNE_CHOICE" = "y" ] && echo "true" || echo "false")
[ "$TUNE" = "false" ] && EPOCHS=$(prompt "Epochs" "${FT_EPOCHS:-1}") && BATCH_SIZE=$(prompt "Batch" "${FT_BATCH_SIZE:-2}") && LR=$(prompt "LR" "${FT_LR:-2e-5}")
UPLOAD_CHOICE=$(prompt "Upload" "$([ "${FT_UPLOAD:-false}" = "true" ] && echo "y" || echo "n")")
UPLOAD=$([ "$UPLOAD_CHOICE" = "y" ] && echo "true" || echo "false")
# The published repo is only ever written by the version-gated publish-model CI
# job. A local upload defaults to a scratch branch on the same repo, which the
# smoke-test cleanup deletes; set FT_REPO_ID to aim elsewhere and FT_REVISION
# to empty to write main (not recommended outside a release).
REPO=$(prompt "Repo" "${FT_REPO_ID:-harpertoken/clue}")
REVISION=$(prompt "Revision (empty means main)" "${FT_REVISION:-ci-smoke}")

# Summary
echo "Config: $DATASET | $TASK | $EPOCHS epochs | $BATCH_SIZE batch | $LR lr | Upload: $UPLOAD to $REPO${REVISION:+ at $REVISION}"

# Confirm
[ "$(prompt "Proceed" "y")" != "y" ] && echo "Cancelled." && exit 0

# Set environment variables for scripts
export FT_DATASET=$DATASET
export FT_TASK=$TASK
export FT_TUNE=$TUNE
export FT_EPOCHS=$EPOCHS
export FT_BATCH_SIZE=$BATCH_SIZE
export FT_LR=$LR
export FT_UPLOAD=$UPLOAD
export FT_REPO_ID=$REPO
export FT_REVISION=$REVISION
export PYTHONPATH=.

# Run training script
echo "Starting training..."
python scripts/train.py

# Run evaluation script
echo "Starting evaluation..."
python scripts/evaluate.py

echo "Build completed successfully!"
