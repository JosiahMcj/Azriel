#!/usr/bin/env bash
# Fuse a LoRA adapter into the base model, producing a STANDALONE
# Azriel model directory that runs without needing the LoRA + base
# loaded separately.
#
# This is the critical step for distribution: after fusing, the
# resulting directory is a complete model (config.json + weights)
# that any mlx_lm/transformers loader can use directly. No further
# adapter juggling.
#
# Usage:
# bash scripts/77_fuse_lora.sh # uses defaults
# ADAPTER=lora-azriel-v0.6.2-delta2 bash scripts/77_fuse_lora.sh
#
# Defaults to fusing v0.6.0 (the current live release).
#
# Output: ~/.azriel/checkpoints/azriel-<tag>-fused/
# Then mlx_lm.generate --model ~/.azriel/checkpoints/azriel-<tag>-fused
# works directly with no --adapter-path argument.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/.azriel}"
VENV_PY="$WORKSPACE/.venv/bin/python"

# Which adapter to fuse. Default: v0.6.0 (live).
# Override with: ADAPTER=lora-azriel-v0.6.2-delta2 bash ...
ADAPTER_NAME="${ADAPTER:-lora-azriel-v0.6.0}"
ADAPTER_PATH="$WORKSPACE/checkpoints/$ADAPTER_NAME"

# Tag derived from adapter name. Strips lora-azriel- prefix.
TAG="${ADAPTER_NAME#lora-azriel-}"
SAVE_PATH="$WORKSPACE/checkpoints/azriel-${TAG}-fused"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

echo "=== Fusing LoRA into base ==="
echo " base: $BASE_MODEL"
echo " adapter: $ADAPTER_PATH"
echo " output: $SAVE_PATH"
echo

if [ ! -x "$VENV_PY" ]; then
  echo "venv python not found at $VENV_PY" >&2; exit 1
fi
if [ ! -d "$ADAPTER_PATH" ]; then
  echo "adapter dir not found: $ADAPTER_PATH" >&2; exit 1
fi
if [ ! -f "$ADAPTER_PATH/adapters.safetensors" ]; then
  echo "adapter weights missing: $ADAPTER_PATH/adapters.safetensors" >&2; exit 1
fi

# Run the fuse. Stays quantized at the base model's bit-width by
# default; pass --dequantize if you want bf16 output (much larger).
"$VENV_PY" -m mlx_lm fuse \
  --model "$BASE_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --save-path "$SAVE_PATH"

echo
echo "=== Fuse done. Verifying standalone load ==="
"$VENV_PY" -m mlx_lm generate \
  --model "$SAVE_PATH" \
  --prompt "Quote Hebrews 11:1." \
  --max-tokens 60 \
  --temp 0.0 || {
    echo "ERROR: fused model failed to load standalone" >&2
    exit 1
  }

echo
echo "=== SUCCESS (bf16 fused) ==="
echo "Standalone model at: $SAVE_PATH"
echo
echo "Total size:"
du -sh "$SAVE_PATH"
echo

# Re-quantize for distribution. The fuse produces bf16 (~57 GB for a
# 30B-A3B model), which is unfriendly for upload + slow to load on
# end-user hardware. A 4-bit re-quant brings it back to ~17 GB and
# matches the format mlx-lm-loaded base models ship in.
if [ "${SKIP_QUANTIZE:-0}" != "1" ]; then
  QUANT_PATH="${SAVE_PATH%-fused}-fused-q4"
  echo "=== Re-quantizing to 4-bit for distribution ==="
  echo " source: $SAVE_PATH"
  echo " output: $QUANT_PATH"
  "$VENV_PY" -m mlx_lm convert \
    --hf-path "$SAVE_PATH" \
    --mlx-path "$QUANT_PATH" \
    -q --q-bits 4 --q-group-size 64 || {
      echo "WARNING: re-quantize failed; bf16 fused model still usable" >&2
    }
  if [ -d "$QUANT_PATH" ]; then
    echo
    echo "Quantized size:"
    du -sh "$QUANT_PATH"
  fi
fi

echo
echo "To run it from anywhere with no other dependency:"
echo " $VENV_PY -m mlx_lm generate \\"
echo " --model $SAVE_PATH \\"
echo " --prompt 'Your question' \\"
echo " --max-tokens 200"
echo
echo "To publish to Hugging Face:"
echo " REPO=JosiahMcj/azriel-${TAG} \\"
echo " AZRIEL_HF_PUBLISH_AUTHORIZED=1 \\"
echo " bash scripts/78_publish_to_hf.sh"
