#!/usr/bin/env bash
# Dry-run extension: 10 iters at lr=2e-5 to confirm the
# iter-5 train-loss bounce was batch=1 noise, not slow divergence.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/.azriel}"
VENV_PY="$WORKSPACE/.venv/bin/python"
LOG_DIR="$WORKSPACE/logs"
LOG="$LOG_DIR/eta3c-dryrun10-$(date +%Y%m%d-%H%M%S).log"
CONFIG="$HOME/azriel-arch/configs/phase_eta_3c_dryrun.yaml"
MODEL="$WORKSPACE/checkpoints/qwen3.6-35b-a3b-mlx-4bit"
DATA_DIR="$WORKSPACE/data/lora_eta3b"
ADAPTER_DIR="$WORKSPACE/checkpoints/lora-azriel-v0.8.0-dryrun10"

mkdir -p "$LOG_DIR" "$ADAPTER_DIR"

set -x
"$VENV_PY" -m mlx_lm.lora -c "$CONFIG" \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_DIR" \
  --num-layers 8 \
  --iters 10 \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --steps-per-report 1 \
  --grad-checkpoint 2>&1 | tee "$LOG"
set +x

echo "=== 10-iter dry-run done -- log: $LOG ==="
