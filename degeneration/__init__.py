"""Degeneration probe: predict 1 - TTR over a forward sliding window of tokens.

Fork-local extension of the hallucination_probes repo. Reuses:
  - probe.value_head_probe.ValueHeadProbe (the hook + linear head)
  - utils.model_utils.setup_lora_for_layers (LoRA wiring)
  - utils.hooks.add_hooks

Adds:
  - DegenerationDataset: loads generations.jsonl and computes per-token
    sliding-window 1 - TTR labels.
  - train(): MSE regression loop over sigmoided probe logits.
  - evaluate(): regression metrics + coarse classification sanity.
"""
