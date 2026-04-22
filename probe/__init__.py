"""Probe module for hallucination detection."""

from .value_head_probe import ValueHeadProbe

# The remaining re-exports pull in heavy hallucination-specific deps (datasets,
# termcolor, dataset_converters, etc.). Import them lazily so lightweight use
# cases (serving-only, or the fork's `degeneration/` subpackage for the TTR
# regression probe) don't require the full dep set.
try:
    from .config import ProbeConfig, TrainingConfig, EvaluationConfig
    from .loss import (
        compute_probe_bce_loss,
        compute_probe_max_aggregation_loss,
        compute_sparsity_loss,
        compute_kl_divergence_loss,
        mask_high_loss_spans,
    )
    from .types import ProbingItem, AnnotatedSpan
    from .dataset import (
        TokenizedProbingDataset,
        tokenized_probing_collate_fn,
        create_probing_dataset,
    )
except ImportError:
    pass

__all__ = [
    "ValueHeadProbe",
    "ProbeConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "compute_probe_bce_loss",
    "compute_probe_max_aggregation_loss",
    "compute_sparsity_loss",
    "compute_kl_divergence_loss",
    "mask_high_loss_spans",
    "setup_probe",
    "ProbingItem",
    "AnnotatedSpan",
    "TokenizedProbingDataset",
    "tokenized_probing_collate_fn",
    "create_probing_dataset",
]
