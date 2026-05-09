#!/usr/bin/env bash
# Full recipe -- gentler recipe LoRA on Qwen3.6-35B-A3B with the
# 10-iter dry-run smoke verified to produce coherent text.
#
# Recipe: rank=16, lr=2e-5, 120 iters, batch=1, num_layers=8,
# max_seq_length=1024, --grad-checkpoint
#
# Output: ~/.azriel/checkpoints/lora-azriel-v0.8.0-pre3/
#
# Hard halt rule: if val loss reverses at any eval (every 15 iters),
# kill the run and use the prior checkpoint (every 30 iters).
#
# Run on a development machine:
# bash ~/azriel-arch/scripts/69_eta_3c_full.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/.azriel}"
VENV_PY="$WORKSPACE/.venv/bin/python"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data/lora_eta3b}"
ADAPTER_DIR="${ADAPTER_DIR:-$WORKSPACE/checkpoints/lora-azriel-v0.8.0-pre3}"
MODEL="${MODEL:-$WORKSPACE/checkpoints/qwen3.6-35b-a3b-mlx-4bit}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/logs}"
LOG="$LOG_DIR/eta3c-train-$(date +%Y%m%d-%H%M%S).log"

LORA_LAYERS="${LORA_LAYERS:-8}"
LORA_RANK="${LORA_RANK:-16}"
LR="${LR:-2e-5}"
ITERS="${ITERS:-120}"
BATCH="${BATCH:-1}"

mkdir -p "$ADAPTER_DIR" "$LOG_DIR"

if [ ! -x "$VENV_PY" ]; then
  echo "venv python not found at $VENV_PY" >&2; exit 1
fi
if ! "$VENV_PY" -c "import mlx_lm" 2>/dev/null; then
  echo "mlx_lm not installed in $VENV_PY -- aborting" >&2; exit 1
fi
if [ ! -d "$MODEL" ]; then
  echo "base model dir not found: $MODEL" >&2; exit 1
fi
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "training data not found: $DATA_DIR/train.jsonl" >&2; exit 1
fi

echo "=== eta.3c full LoRA train ==="
echo " base: $MODEL"
echo " data: $DATA_DIR (presplit)"
echo " adapter: $ADAPTER_DIR"
echo " log: $LOG"
echo " rank=$LORA_RANK layers=$LORA_LAYERS lr=$LR iters=$ITERS batch=$BATCH"
echo

CONFIG="${CONFIG:-$HOME/azriel-arch/configs/phase_eta_3c_full.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG" >&2; exit 1
fi

set -x
"$VENV_PY" -m mlx_lm.lora -c "$CONFIG" \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_DIR" \
  --num-layers "$LORA_LAYERS" \
  --iters "$ITERS" \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --grad-checkpoint \
  2>&1 | tee "$LOG"
set +x

echo
echo "=== eta.3c train done ==="
echo " adapter saved to: $ADAPTER_DIR"
