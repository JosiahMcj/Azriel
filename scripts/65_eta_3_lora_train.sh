#!/usr/bin/env bash
# mini-recipe LoRA on Qwen3.6-35B-A3B base.
#
# Recipe (per ROADMAP_POST_ZETA.md η.3):
# rank=16, lr=1e-4, 30 iters, batch=1, on the v0.6.0 corpus
#
# Output: ~/.azriel/checkpoints/lora-azriel-v0.8.0-pre/
#
# This is a vanilla LoRA train -- (looped middle layers + LTI)
# is applied at INFERENCE time via load_phase_beta, not during training.
# Confirming works on Qwen3.6 is a separate post-train check.
#
# Hard halt rule from η.3 spec: if Phase 1 safety probe drops below 8/8
# even once, training is rejected, REJECTED.md drop, symlink unchanged.
#
# Run on a development machine:
# bash ~/azriel-arch/scripts/65_eta_3_lora_train.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/.azriel}"
VENV_PY="$WORKSPACE/.venv/bin/python"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data/lora}"
ADAPTER_DIR="${ADAPTER_DIR:-$WORKSPACE/checkpoints/lora-azriel-v0.8.0-pre}"
MODEL="${MODEL:-$WORKSPACE/checkpoints/qwen3.6-35b-a3b-mlx-4bit}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/logs}"
LOG="$LOG_DIR/eta3-train-$(date +%Y%m%d-%H%M%S).log"

# η.3 mini-recipe knobs (from PHASE_ETA_ENTRY_POINT.md)
LORA_LAYERS="${LORA_LAYERS:-8}"
LORA_RANK="${LORA_RANK:-16}"
LR="${LR:-1e-4}"
ITERS="${ITERS:-30}"
BATCH="${BATCH:-1}"

mkdir -p "$ADAPTER_DIR" "$LOG_DIR"

if [ ! -x "$VENV_PY" ]; then
  echo "venv python not found at $VENV_PY" >&2
  exit 1
fi
if ! "$VENV_PY" -c "import mlx_lm" 2>/dev/null; then
  echo "mlx_lm not installed in $VENV_PY -- aborting" >&2
  exit 1
fi
if [ ! -d "$MODEL" ]; then
  echo "base model dir not found: $MODEL" >&2
  exit 1
fi
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "training data not found: $DATA_DIR/train.jsonl" >&2
  exit 1
fi

echo "=== η.3 mini-recipe LoRA train ==="
echo " base: $MODEL"
echo " data: $DATA_DIR"
echo " adapter: $ADAPTER_DIR"
echo " log: $LOG"
echo " rank=$LORA_RANK layers=$LORA_LAYERS lr=$LR iters=$ITERS batch=$BATCH"
echo

CONFIG="${CONFIG:-$HOME/azriel-arch/configs/phase_eta_3_lora.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

set -x
# YAML carries the LoRA rank (mlx_lm.lora has no --rank CLI flag).
# CLI args passed here OVERRIDE the YAML where they overlap; the
# explicit ones below mirror the YAML values for clarity in the log.
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
echo "=== η.3 train done ==="
echo " adapter saved to: $ADAPTER_DIR"
echo
echo "Next: 8/8 safety probe via load_phase_beta. NOTE: needs "
echo "compatibility verification on Qwen3.6 before the probe will run."
