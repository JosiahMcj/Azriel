#!/usr/bin/env bash
# Publish a fused Azriel model to Hugging Face.
#
# REQUIRES:
# 1. A fused model directory (run scripts/77_fuse_lora.sh first)
# 2. A Hugging Face token with write permissions to the target repo:
# hf auth login (interactive)
# OR set HF_TOKEN environment variable
# 3. Explicit user authorization. This is a destructive PUBLIC action;
# this script will not run until you remove the safety guard
# block at the top.
#
# Usage:
# REPO=JosiahMcj/azriel-v0.6.0 bash scripts/78_publish_to_hf.sh
# ADAPTER=lora-azriel-v0.6.2-delta2 REPO=JosiahMcj/azriel-v0.6.2-delta2 \
# bash scripts/78_publish_to_hf.sh
#
# What gets uploaded:
# - The fused model directory (config.json, weights, tokenizer files)
# - The MODEL_CARD.md from this repo (filled in for the chosen tag)
set -euo pipefail

# === Safety guard ===
if [ "${AZRIEL_HF_PUBLISH_AUTHORIZED:-0}" != "1" ]; then
  cat >&2 <<MSG
ERROR: HF publish is gated. To proceed:

  1. Have a fused model directory ready:
       bash scripts/77_fuse_lora.sh

  2. Set up Hugging Face credentials (one-time):
       hf auth login
     OR
       export HF_TOKEN=hf_your_token_here

  3. Re-run with the safety env var set:
       REPO=JosiahMcj/azriel-v0.6.0 \\
       AZRIEL_HF_PUBLISH_AUTHORIZED=1 \\
       bash scripts/78_publish_to_hf.sh

The repo \$REPO must exist (or you must have create permission).
MSG
  exit 2
fi

WORKSPACE="${WORKSPACE:-$HOME/.azriel}"
VENV_PY="$WORKSPACE/.venv/bin/python"

ADAPTER_NAME="${ADAPTER:-lora-azriel-v0.6.0}"
TAG="${ADAPTER_NAME#lora-azriel-}"
MODEL_DIR="$WORKSPACE/checkpoints/azriel-${TAG}-fused"
REPO="${REPO:-}"

if [ -z "$REPO" ]; then
  echo "ERROR: set REPO=user/azriel-vN.M" >&2
  exit 2
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: fused model not found: $MODEL_DIR" >&2
  echo "Run: bash scripts/77_fuse_lora.sh first" >&2
  exit 2
fi

echo "=== Publishing $TAG to $REPO ==="
echo " source: $MODEL_DIR"
echo " size: $(du -sh "$MODEL_DIR" | cut -f1)"
echo

# Render the model card with the chosen tag.
CARD_TEMPLATE="$HOME/azriel-arch/docs/MODEL_CARD_TEMPLATE.md"
if [ ! -f "$CARD_TEMPLATE" ]; then
  CARD_TEMPLATE="$(dirname "$0")/../docs/MODEL_CARD_TEMPLATE.md"
fi
RENDERED_CARD="$MODEL_DIR/README.md"
if [ -f "$CARD_TEMPLATE" ]; then
  sed "s/\${TAG}/$TAG/g; s|\${REPO}|$REPO|g" "$CARD_TEMPLATE" > "$RENDERED_CARD"
  echo " card: $RENDERED_CARD ($(wc -c < "$RENDERED_CARD") bytes)"
else
  echo " WARNING: model card template not found; uploading without README" >&2
fi

# Bundle Apache 2.0 LICENSE into the model dir (Apache 2.0 obligation
# for derivative works -- audit verified Qwen3-Coder ships vanilla
# Apache 2.0 with no NOTICE file). Pull from the upstream HF repo so
# the bundled text is byte-identical to what Qwen distributes.
LICENSE_OUT="$MODEL_DIR/LICENSE"
if [ ! -f "$LICENSE_OUT" ]; then
  echo " fetching upstream Apache 2.0 LICENSE..."
  if curl -fsSL "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/raw/main/LICENSE" \
        -o "$LICENSE_OUT" 2>/dev/null; then
    echo " license: $LICENSE_OUT ($(wc -c < "$LICENSE_OUT") bytes)"
  else
    echo " WARNING: could not fetch upstream LICENSE; aborting publish" >&2
    exit 3
  fi
fi

# NOTICE file documenting the derivative chain. Written every publish
# run so any tag/repo update is visible in the bundled NOTICE.
cat > "$MODEL_DIR/NOTICE" <<NOTICE_EOF
Azriel ${TAG} -- a fine-tuned derivative of Qwen3-Coder-30B-A3B-Instruct.

Base model:
  Name: Qwen/Qwen3-Coder-30B-A3B-Instruct
  Author: Alibaba Cloud
  License: Apache License 2.0
  Source: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct

Modifications:
  - LoRA fine-tune (rank 16, scale 20, dropout 0.05, last 16 transformer
    blocks) merged into the base weights.
  - Re-quantized to 4-bit (q-group-size 64) for distribution.
  - Training data + runtime code are author-original work
    (https://github.com/JosiahMcj/Azriel).

This entire derivative is distributed under the Apache License 2.0
(see LICENSE). All upstream attribution is preserved.
NOTICE_EOF
echo " notice: $MODEL_DIR/NOTICE ($(wc -c < "$MODEL_DIR/NOTICE") bytes)"
echo

# huggingface_hub upload-folder is the right primitive for a model dir.
# This both creates the repo (if missing + token has create perms) and
# uploads everything in MODEL_DIR.
"$VENV_PY" -m huggingface_hub upload \
  "$REPO" \
  "$MODEL_DIR" \
  . \
  --repo-type model \
  --commit-message "Publish azriel-${TAG} (fused LoRA into Qwen3-Coder-30B-A3B-Instruct)"

echo
echo "=== SUCCESS ==="
echo "Published at: https://huggingface.co/$REPO"
echo
echo "Anyone can now run Azriel with no LoRA juggling:"
echo " pip install mlx-lm"
echo " mlx_lm.generate --model $REPO --prompt 'Your question'"
