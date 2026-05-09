"""v0.6.0-pre: continue fine-tune of the v0.5 LoRA under the looped
AzrielModel wrapper.

Replicates mlx_lm.lora.train_model's freeze + linear_to_lora_layers pattern
exactly so the trainable parameter set matches v0.5's training (rank=8,
num_layers=16). Only difference: the model is wrapped in AzrielModel
before training.

Outputs to: ~/.azriel/checkpoints/lora-azriel-v0.6.0/
"""
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as opt
from mlx_lm import load
from mlx_lm.tuner import datasets, trainer
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from azriel import AzrielConfig, AzrielModel

HOME = Path.home()
BASE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
V05_ADAPTER = HOME / ".azriel" / "checkpoints" / "lora-azriel-v0.6.0"
OUT_DIR = HOME / ".azriel" / "checkpoints" / "lora-azriel-v0.7.0-pre"
DATA_DIR = HOME / ".azriel" / "data" / "lora"

ITERS = 100
BATCH = 1
MAX_SEQ = 1024
LR = 1e-5
LTI_EXPANSION = 1
# Align the looped slice with v0.5's LoRA region (last 16 of 48 layers). When
# the loop slice sits OUTSIDE the LoRA region, adding a trainable LTI on top
# forces backprop through the extra frozen layers above the LoRA, blowing
# memory. With slice == LoRA region, backward exactly matches v0.5's
# established profile (~65GB peak at seq=2048) plus tiny LTI overhead.
LOOP_START = 32
LOOP_END = 48


def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading base only (no adapter)", flush=True)
    base, tokenizer = load(BASE_MODEL)

    src_cfg = json.loads((V05_ADAPTER / "adapter_config.json").read_text())
    num_layers = src_cfg["num_layers"]
    lora_params = src_cfg["lora_parameters"]
    print(f"v0.5 config: num_layers={num_layers}, lora_parameters={lora_params}", flush=True)

    # Canonical mlx_lm.lora pattern: freeze base, then linear_to_lora_layers
    # constructs LoRALinear modules with the inner base Linear FROZEN and only
    # the lora deltas TRAINABLE.
    base.freeze()
    linear_to_lora_layers(base, num_layers, lora_params, use_dora=False)
    # Load the v0.5-candidate weights into the freshly-attached LoRA modules.
    base.load_weights(str(V05_ADAPTER / "adapters.safetensors"), strict=False)

    # Real heavy layers run once, LTI iterates once on top. With LTI
    # zero-init the wrapper is logit-identical to base at init, so the v0.5
    # identity is preserved and training only adds whatever LTI learns to do.
    cfg = AzrielConfig(
        loop_layer_start=LOOP_START,
        loop_layer_end=LOOP_END,
        loop_max_iters=2,
        lti_enabled=True,
        lti_expansion=LTI_EXPANSION,
    )
    print(f"wrapping: {cfg}", flush=True)
    model = AzrielModel(base, cfg)
    print_trainable_parameters(base)

    print(f"[{time.strftime('%H:%M:%S')}] loading dataset from {DATA_DIR}", flush=True)
    args_obj = type("A", (), {
        "data": str(DATA_DIR),
        "train": True, "test": False,
        "hf_dataset": None, "config": {},
        "chat_template": None,
        "mask_prompt": False, # match v0.5
    })()
    train_ds, valid_ds, _ = datasets.load_dataset(args_obj, tokenizer)
    train_ds = CacheDataset(train_ds)
    valid_ds = CacheDataset(valid_ds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_file = OUT_DIR / "adapters.safetensors"
    (OUT_DIR / "adapter_config.json").write_text((V05_ADAPTER / "adapter_config.json").read_text())

    train_args = trainer.TrainingArgs(
        batch_size=BATCH,
        iters=ITERS,
        val_batches=5,
        steps_per_report=10,
        steps_per_eval=200,
        steps_per_save=25,
        max_seq_length=MAX_SEQ,
        adapter_file=str(adapter_file),
        grad_checkpoint=True,
    )

    optimizer = opt.AdamW(learning_rate=LR)

    print(f"[{time.strftime('%H:%M:%S')}] starting training: "
          f"{ITERS} iters, batch={BATCH}, max_seq={MAX_SEQ}, lr={LR}", flush=True)
    t0 = time.time()
    trainer.train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=valid_ds,
        args=train_args,
    )
    print(f"[{time.strftime('%H:%M:%S')}] training done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
