#!/usr/bin/env bash
# Dry-run: 5 iters at lr=2e-5 to check gradient stability
# on Qwen3.6-35B-A3B before committing to a full eta.3c run.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/.azriel}"
VENV_PY="$WORKSPACE/.venv/bin/python"
LOG_DIR="$WORKSPACE/logs"
LOG="$LOG_DIR/eta3c-dryrun-$(date +%Y%m%d-%H%M%S).log"
CONFIG="$HOME/azriel-arch/configs/phase_eta_3c_dryrun.yaml"
MODEL="$WORKSPACE/checkpoints/qwen3.6-35b-a3b-mlx-4bit"
DATA_DIR="$WORKSPACE/data/lora_eta3b"
ADAPTER_DIR="$WORKSPACE/checkpoints/lora-azriel-v0.8.0-dryrun"

mkdir -p "$LOG_DIR" "$ADAPTER_DIR"

set -x
"$VENV_PY" -m mlx_lm.lora -c "$CONFIG" \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_DIR" \
  --num-layers 8 \
  --iters 5 \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --grad-checkpoint 2>&1 | tee "$LOG"
set +x

echo "=== dry-run done -- log: $LOG ==="
