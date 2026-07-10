# MCED v5.1 — Non-bisulfite-based image classification for early cancer detection

This repository is the **inference pipeline for the MCED v5.1 model**. It takes a sample ID,
retrieves its corresponding image data, generates embeddings using multiple pretrained image
classification models, and performs the final prediction using a stacked machine learning model.

---

## Overview

- **Retrieve Input**: Fetches the sample data from the `--data_dir` passed on the command line.
- **Embedding Generation**: Uses multiple pretrained image classification models to extract embeddings.
- **Model Stacking**: Concatenates embeddings and feeds them into a trained stacking model.
- **Final Prediction**: Returns the output probability / prediction result.

---

## Repository Structure

```
.
├── config/                     # Model configuration (mced_v5.1)
├── model_repository/           # Model checkpoints (mced_v5.1)
├── metadata/                   # Sample request file format
├── run/                        # Example run script
└── src                         # Source code
    ├── data                    # Define how to load the data from local directory
    │   ├── __init__.py
    │   ├── load.py
    │   └── utils.py
    ├── schemas.py               # Input/output schemas
    ├── model                    # Define how to load checkpoints and create embeddings
    │   ├── _base.py
    │   ├── keras.py
    │   ├── one_dim.py
    │   └── torch.py
    └── stacking                 # Define how to stack the base models
        ├── __init__.py
        ├── _base.py
        ├── binary_task.py
        └── muliclass_task.py
```

---

## Installation

### Method 1: Manual Setup with `make`

```bash
conda create --name ecd python=3.9 -y
conda activate ecd
make
```

> Installs Python dependencies and prepares the environment based on `Makefile`.

### Method 2: Using Docker

```bash
docker compose build
docker compose run --rm -it nonbs-stacking python run/pred_mced_v51.py \
  --data_dir /path/to/data_dir \
  --metadata /path/to/metadata.csv \
  --save_dir /path/to/save_dir
```

---

## Inference Usage

```python
from omegaconf import OmegaConf

from src import BinaryStacking

config = OmegaConf.load("config/mced_stacking_resnet_and_single_task_cnn_v51.yaml")

model = BinaryStacking(
    config=config,
    data_dir="/path/to/data_dir",
)

RUN_ID = "RUN_ID"
SAMPLE_ID = "SAMPLE_ID"

pred = model(run_id=RUN_ID, sample_id=SAMPLE_ID)
print(pred.model_dump())

# ModelPrediction(prob=0.020376254700762868, pred_id=0)
```

`metadata/test_request.csv` shows the expected metadata format (`NONBS_ID`, `Run_NONBS` columns)
for batch scoring via `run/pred_mced_v51.py`.

---

## Model weights

Model checkpoints live under `model_repository/mced_stacking_resnet_and_single_task_cnn_v51/`.
This folder is **not tracked by git** — only its structure (`.gitkeep`) is. Do not force-add or
push the checkpoint files (`*.pkl`, `*.pt`, `*.keras`) to any remote repository.

---

## Verified

This handover was smoke-tested end to end (config load → all 6 checkpoints load → embeddings
→ stacking classifier → final probability) using synthetic feature CSVs, since real patient/
commercial samples are intentionally not shipped in this repo. Confirmed working:

- `config/mced_stacking_resnet_and_single_task_cnn_v51.yaml` loads and resolves all 6 checkpoint
  paths under `model_repository/`.
- The stacking model (`mced_stacking_single_task_cnn_and_resnet_v51.pkl`), the torch checkpoint
  (`resnet_fnuc_flen.pt`), and all 5 `.keras` checkpoints load and run a forward pass.
- `BinaryStacking.__call__` runs the full pipeline (feature loading → embedding → stacking
  prediction) without errors.
- `scikit-learn==1.6.1` is pinned in `requirements.txt` because the `.pkl` was serialized with
  that version; other versions raise `InconsistentVersionWarning` (usually harmless, but pin it
  to avoid future breakage).
- `config/motif/*.csv` (generic k-mer ordering tables) are required by `src/data/utils.py` for
  the 3-mer downgrade path (`singletask_cnn_em_flen_3mer`) — make sure this folder stays alongside
  the config/model.

Not verified: prediction correctness/accuracy — that requires real sample feature data, which is
out of scope for this code handover.
