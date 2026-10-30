# OpticalRAG

## Installation

Runtime environment: Linux, Python 3.10, and PyTorch 2.8.0 with CUDA 12.8. Full inference requires a CUDA GPU.

```bash
conda create -n opticalrag python=3.10 -y
conda activate opticalrag
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Prepare Assets

Models, datasets, and visual caches require substantial storage. Place the asset directory on a drive with sufficient free space. The download script uses pinned model and dataset revisions. Inference uses local files without downloading assets automatically.

```bash
export ASSET_DIR=/path/to/large-disk/opticalrag-assets
python scripts/download_assets.py --assets "$ASSET_DIR"
```

The trained compressor/bridge checkpoint is located at `checkpoints/checkpoint.pt`. The script above downloads the base models and evaluation data. Supply the font separately; the pipeline generates the visual cache.

- Compressor checkpoint: contains the `qcompress` and `bridge` state dictionaries. SHA256: `0a2d65b03785f2e7d77df194799eb237ced7ce7309ed7e47462254ccb44f985c`.
- Font: `DejaVuSansMono.ttf`. SHA256: `cf3584503be005b9790678091386191b83e3cfa77ae93d179dbe8f8ce6264013`. Different font versions may change pagination and visual features.
- Pass the checkpoint and font paths as command-line arguments. The pipeline strictly validates their hashes.

## Run the Pipeline

```bash
export CHECKPOINT=./checkpoints/checkpoint.pt
export FONT=/path/to/DejaVuSansMono.ttf

# Smoke test with a separate cache; this is not a full evaluation
python scripts/run_pipeline.py --assets "$ASSET_DIR" --checkpoint "$CHECKPOINT" \
  --font "$FONT" --run-id smoke_001 --limit 2

# Full pipeline: cache -> FAISS index -> evaluation on 503 samples -> audit
python scripts/run_pipeline.py --assets "$ASSET_DIR" --checkpoint "$CHECKPOINT" \
  --font "$FONT" --run-id full_001

# Rerun evaluation using the full cache; use a new run ID each time
python scripts/run_pipeline.py --assets "$ASSET_DIR" --checkpoint "$CHECKPOINT" \
  --font "$FONT" --run-id repeat_001 --stage eval
```

Use `--stage cache|index|eval|all` to select a pipeline stage and `--cache /path/to/cache` to specify an existing cache. Avoid concurrent writes to the same cache. Use `--plan-only` to print commands without loading models.

Outputs are saved to `runs/<run-id>/`, including configuration, logs, full predictions, a summary, execution status, and an independent audit. The pipeline refuses to overwrite an existing run directory. Full predictions include the original contexts; handle them according to the data source's terms.

## Sources and Licensing

This project does not currently include a license for its own code. Models and datasets remain subject to their respective licenses:

- Qwen2.5-7B-Instruct: `Qwen/Qwen2.5-7B-Instruct`.
- DeepSeek-OCR: `deepseek-ai/DeepSeek-OCR`. Model definitions are loaded from a pinned revision.
- LongBench-v2: `zai-org/LongBench-v2`. See `configs/longbench.json` for the pinned revision.
- The font must be supplied by the user and remains subject to its original license.

This repository provides inference and data pipeline code. It does not include a training workflow.
