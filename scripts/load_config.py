import os
import shlex
import sys
import yaml

# Load config from config.yaml if exists
config_file = "config.yaml"
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

# Set env vars with defaults
os.environ["FT_DATASET"] = config.get("dataset", "squad")
os.environ["FT_TASK"] = config.get("task", "qa")
os.environ["FT_EPOCHS"] = str(config.get("epochs", 1))
os.environ["FT_BATCH_SIZE"] = str(config.get("batch_size", 2))
os.environ["FT_LR"] = str(config.get("learning_rate", 2e-5))
os.environ["FT_UPLOAD"] = str(config.get("upload", False)).lower()
os.environ["FT_TUNE"] = str(config.get("tune", False)).lower()
hf_token = config.get("hf_token", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

if "--export" in sys.argv:
    # Print shell assignments for run.sh to eval. Setting os.environ above only
    # affects this process, so a bare `python3 scripts/load_config.py` call never
    # reached the caller and config.yaml was silently inert.
    for key in (
        "FT_DATASET",
        "FT_TASK",
        "FT_EPOCHS",
        "FT_BATCH_SIZE",
        "FT_LR",
        "FT_UPLOAD",
        "FT_TUNE",
        "HF_TOKEN",
    ):
        if key in os.environ:
            print(f"export {key}={shlex.quote(os.environ[key])}")
else:
    print("Config loaded.")
