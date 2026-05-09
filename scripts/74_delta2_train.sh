#!/usr/bin/env bash
# Train v0.6.2-delta2 LoRA on the teacher-distilled corpus.
#
# Recipe: rank=32, lr=5e-5, 200 iters, batch=1, num_layers=16,
# max_seq_length=2048
# Output: ~/.azriel/checkpoints/lora-azriel-v0.6.2-delta2/
#
# Hard halt: 8/8 safety probe must hold. <8/8 = REJECTED, symlink unchanged.
#
# Run on a development machine:
# bash ~/azriel-arch/scripts/74_delta2_train.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/.azriel}"
VENV_PY="$WORKSPACE/.venv/bin/python"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data/delta2}"
ADAPTER_DIR="${ADAPTER_DIR:-$WORKSPACE/checkpoints/lora-azriel-v0.6.2-delta2}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/logs}"
LOG="$LOG_DIR/delta2-train-$(date +%Y%m%d-%H%M%S).log"

LORA_LAYERS="${LORA_LAYERS:-8}"
LORA_RANK="${LORA_RANK:-16}"
LR="${LR:-5e-5}"
ITERS="${ITERS:-200}"
BATCH="${BATCH:-1}"

mkdir -p "$ADAPTER_DIR" "$LOG_DIR"

if [ ! -x "$VENV_PY" ]; then
  echo "venv python not found at $VENV_PY" >&2; exit 1
fi
# MODEL may be either a local dir OR a HuggingFace repo id
# (e.g. "Qwen/Qwen3-Coder-30B-A3B-Instruct"). mlx_lm.lora handles
# both; we only validate the local-path case.
if [[ "$MODEL" == /* || "$MODEL" == "$HOME"* ]]; then
  if [ ! -d "$MODEL" ]; then
    echo "base model dir not found: $MODEL" >&2; exit 1
  fi
fi
# delta.2.5 expects mlx_lm chat format with messages: [system, user, assistant]
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "training data not found: $DATA_DIR/train.jsonl" >&2
  echo "Run delta.2.4 (scripts/73) first AND copy the output to data/delta2/{train,valid}.jsonl" >&2
  exit 1
fi

echo "=== delta.2.5 LoRA train ==="
echo " base: $MODEL"
echo " data: $DATA_DIR"
echo " adapter: $ADAPTER_DIR"
echo " log: $LOG"
echo " rank=$LORA_RANK layers=$LORA_LAYERS lr=$LR iters=$ITERS batch=$BATCH"
echo

CONFIG="${CONFIG:-$HOME/azriel-arch/configs/phase_delta2_lora.yaml}"
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
echo "=== delta.2.5 done. Adapter at: $ADAPTER_DIR ==="
echo "Next: scripts/75 (delta.2.6 side-by-side probe vs v0.6.0)"
