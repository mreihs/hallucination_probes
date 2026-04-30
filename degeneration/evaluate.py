"""Evaluation for the degeneration regression probe."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from probe.value_head_probe import ValueHeadProbe

from .data_loader import DegenerationDataset, make_collate_fn

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_regression(
    probe: ValueHeadProbe,
    loader: DataLoader,
) -> Dict[str, float]:
    """
    Run the probe and compute regression metrics against the masked targets.

    Returns:
        mse:         mean squared error over labelled positions
        mae:         mean absolute error over labelled positions
        pearson:     Pearson correlation between predictions and labels
        auc_at_0_5:  classification AUC where targets are binarised at 0.5
                     (coarse sanity check — "does the probe separate low vs
                     high repetition regions at all").
    """
    probe.eval()
    device = next(probe.parameters()).device

    preds_all: list[float] = []
    labels_all: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)

        out = probe(input_ids=input_ids, attention_mask=attention_mask)
        probe_logits = out["probe_logits"].squeeze(-1)
        preds = torch.sigmoid(probe_logits)

        mask = label_mask.bool()
        preds_all.extend(preds[mask].detach().cpu().tolist())
        labels_all.extend(labels[mask].detach().cpu().tolist())

    if not labels_all:
        return {"mse": float("nan"), "mae": float("nan"),
                "pearson": float("nan"), "auc_at_0_5": float("nan"),
                "n_tokens": 0}

    preds_np = np.asarray(preds_all, dtype=np.float64)
    labels_np = np.asarray(labels_all, dtype=np.float64)

    mse = float(np.mean((preds_np - labels_np) ** 2))
    mae = float(np.mean(np.abs(preds_np - labels_np)))

    # Pearson; guarded for zero-variance edge cases.
    if preds_np.std() < 1e-12 or labels_np.std() < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(preds_np, labels_np)[0, 1])

    # Coarse classification AUC: binarise labels at 0.5.
    auc = float("nan")
    binary = (labels_np >= 0.5).astype(np.float64)
    if 0 < binary.sum() < len(binary):
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(binary, preds_np))
        except Exception as e:  # pragma: no cover
            log.warning("roc_auc_score failed: %s", e)

    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
        "auc_at_0_5": auc,
        "n_tokens": int(len(labels_np)),
    }


# -------------------------------------------------------------------
# CLI-style entry: load a checkpoint, evaluate on JSONL, save metrics.
# -------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    eval_data: str | Path,
    *,
    batch_size: int = 4,
    max_length: int = 2048,
    output_dir: Optional[str | Path] = None,
    model_name: Optional[str] = None,
) -> Dict[str, float]:
    from peft import PeftModel
    from utils.model_utils import load_model_and_tokenizer

    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "degeneration_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"degeneration_meta.json not found in {checkpoint_dir}; "
            f"was this checkpoint produced by degeneration.train?"
        )
    meta = json.loads(meta_path.read_text())

    resolved_model = model_name or meta.get("model_name")
    if resolved_model is None:
        raise ValueError(
            f"Could not infer model_name from {checkpoint_dir}; "
            f"pass model_name explicitly."
        )

    model, tokenizer = load_model_and_tokenizer(resolved_model)
    if (checkpoint_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    probe = ValueHeadProbe(model, path=checkpoint_dir)

    ds = DegenerationDataset(eval_data)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(
            tokenizer,
            max_length=max_length,
            window_size=int(meta.get("window_size", 256)),
            primary_n=int(meta.get("primary_n", 1)),
            ttr_threshold=meta.get("ttr_threshold"),
            smoothing=bool(meta.get("label_smoothing", False)),
        ),
    )

    metrics = evaluate_regression(probe, loader)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
