"""Serving-time probe loading: base model → LoRA adapter → ValueHeadProbe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from peft import PeftModel

from probe.value_head_probe import ValueHeadProbe


def load_probe(
    checkpoint_dir: str | Path,
    base_model,
) -> ValueHeadProbe:
    """
    Attach LoRA adapters (if the checkpoint has them) and the saved probe
    head to `base_model`, returning a ready-to-use ValueHeadProbe.

    The worker's `GenerationEngine` can register a forward hook on
    `probe.target_module` and read per-token logits via `probe.value_head`
    without invoking `probe.forward()`.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if (checkpoint_dir / "adapter_config.json").exists():
        base_model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))

    probe = ValueHeadProbe(base_model, path=checkpoint_dir)
    return probe


def read_probe_config(checkpoint_dir: str | Path) -> dict:
    """Convenience: read probe_config.json from a checkpoint dir."""
    return json.loads(
        (Path(checkpoint_dir) / "probe_config.json").read_text()
    )
