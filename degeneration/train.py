"""Training entry point for the degeneration probe (MSE regression on 1 - TTR)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

# Fork-native imports (present on sys.path when the fork root is first in it).
from probe.value_head_probe import ValueHeadProbe
from utils.model_utils import (
    load_model_and_tokenizer,
    setup_lora_for_layers,
    get_num_layers,
    print_trainable_parameters,
)

from .data_loader import DegenerationDataset, make_collate_fn
from .evaluate import evaluate_regression


log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Model + probe
    model_name: str
    layer: int
    lora_enabled: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # Data
    train_data: List[str] = field(default_factory=list)
    eval_data: Optional[str] = None
    eval_fraction: float = 0.2
    max_length: int = 2048

    # Label construction
    window_size: int = 256
    primary_n: int = 1

    # Optim
    head_lr: float = 5.0e-3
    lora_lr: float = 5.0e-5
    batch_size: int = 4
    num_epochs: int = 10
    seed: int = 42

    # I/O + logging
    output_dir: str = "outputs/probes"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


# -------------------------------------------------------------------
# Optimiser: split params into head vs LoRA groups with separate LRs.
# Mirrors the pattern in probe/trainer.py::ProbeTrainer.create_optimizer
# but as a plain function (no HF Trainer subclass).
# -------------------------------------------------------------------


def _build_optimizer(probe: ValueHeadProbe, cfg: TrainConfig) -> AdamW:
    head_params, lora_params, other_params = [], [], []
    for name, param in probe.named_parameters():
        if not param.requires_grad:
            continue
        if "value_head" in name:
            head_params.append(param)
        elif "lora" in name.lower():
            lora_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": cfg.head_lr, "name": "head"})
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.lora_lr, "name": "lora"})
    if other_params:
        groups.append({"params": other_params, "lr": cfg.head_lr, "name": "other"})
    return AdamW(groups)


# -------------------------------------------------------------------
# Loss: MSE between sigmoid(probe_logit) and the [0,1] TTR label, masked
# to positions where label_mask == 1.
# -------------------------------------------------------------------


def _masked_mse(
    probe_logits: torch.Tensor,  # [B, T]
    labels: torch.Tensor,        # [B, T]
    label_mask: torch.Tensor,    # [B, T]
) -> torch.Tensor:
    preds = torch.sigmoid(probe_logits)
    sq = (preds - labels) ** 2
    denom = label_mask.sum().clamp(min=1.0)
    return (sq * label_mask).sum() / denom


# -------------------------------------------------------------------
# Main training function
# -------------------------------------------------------------------


def _resolve_dtype() -> torch.dtype:
    """Pick a training dtype that's stable on the current device.

    CPU and MPS: float32 (fp16/bf16 on MPS is flaky, fp16 on CPU is slow).
    CUDA: bfloat16 when supported.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def train(cfg: TrainConfig) -> Path:
    """Run training; return the path to the saved checkpoint directory."""
    torch.manual_seed(cfg.seed)

    log.info("Loading model %s", cfg.model_name)
    model, tokenizer = load_model_and_tokenizer(
        cfg.model_name, torch_dtype=_resolve_dtype(),
    )
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    # Resolve layer (support negative indexing).
    num_layers = get_num_layers(model)
    layer = cfg.layer if cfg.layer >= 0 else num_layers + cfg.layer
    if not 0 <= layer < num_layers:
        raise ValueError(f"layer {cfg.layer} out of range for {num_layers}-layer model")

    # Freeze base model weights. LoRA params are unfrozen by get_peft_model.
    for p in model.parameters():
        p.requires_grad = False

    if cfg.lora_enabled:
        lora_layer_indices = list(range(layer + 1))
        log.info(
            "Attaching LoRA (r=%d, alpha=%d) to layers %d..%d",
            cfg.lora_rank, cfg.lora_alpha, 0, layer,
        )
        model = setup_lora_for_layers(
            model,
            lora_layer_indices,
            lora_r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        )

    probe = ValueHeadProbe(model, layer_idx=layer)
    print_trainable_parameters(probe)

    # --- dataset ---------------------------------------------------
    log.info("Loading train data: %s", cfg.train_data)
    full_ds = DegenerationDataset(cfg.train_data)
    collate = make_collate_fn(
        tokenizer,
        max_length=cfg.max_length,
        window_size=cfg.window_size,
        primary_n=cfg.primary_n,
    )
    if cfg.eval_data is not None:
        train_ds = full_ds
        eval_ds = DegenerationDataset(cfg.eval_data)
    else:
        n_eval = max(1, int(len(full_ds) * cfg.eval_fraction))
        n_train = len(full_ds) - n_eval
        train_ds, eval_ds = random_split(
            full_ds,
            [n_train, n_eval],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    log.info("Train: %d items | Eval: %d items", len(train_ds), len(eval_ds))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate,
    )

    # --- optim + wandb --------------------------------------------
    optimizer = _build_optimizer(probe, cfg)
    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=asdict(cfg),
        )

    device = next(probe.parameters()).device
    global_step = 0

    for epoch in range(cfg.num_epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            out = probe(input_ids=input_ids, attention_mask=attention_mask)
            probe_logits = out["probe_logits"].squeeze(-1)  # [B, T]

            loss = _masked_mse(probe_logits, labels, label_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if use_wandb:
                import wandb
                wandb.log({"train/mse": loss.item(), "train/step": global_step})

        avg = epoch_loss / max(n_batches, 1)
        log.info("Epoch %d/%d — train MSE: %.5f", epoch + 1, cfg.num_epochs, avg)
        if use_wandb:
            import wandb
            wandb.log({"train/epoch_mse": avg, "epoch": epoch + 1})

        # --- per-epoch eval ---------------------------------------
        metrics = evaluate_regression(probe, eval_loader)
        log.info("  eval — mse=%.5f  pearson=%.3f  auc@0.5=%.3f",
                 metrics["mse"], metrics["pearson"], metrics["auc_at_0_5"])
        if use_wandb:
            import wandb
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

    # --- save ------------------------------------------------------
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / ts
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Capture model_name alongside the probe so evaluate / serve can re-load.
    train_snapshot = asdict(cfg)
    (out_dir / "config.json").write_text(
        json.dumps(train_snapshot, indent=2, default=str)
    )

    probe.save(ckpt_dir)
    # Write our extra metadata to a sidecar file so we don't mutate the
    # fork's probe_config.json schema.
    (ckpt_dir / "degeneration_meta.json").write_text(
        json.dumps(
            {
                "model_name": cfg.model_name,
                "lora_enabled": cfg.lora_enabled,
                "window_size": cfg.window_size,
                "primary_n": cfg.primary_n,
            },
            indent=2,
        )
    )

    if use_wandb:
        import wandb
        wandb.finish()

    log.info("Saved probe to %s", ckpt_dir)
    return ckpt_dir
