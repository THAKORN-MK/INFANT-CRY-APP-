"""Leakage-safe shared experiment orchestration primitives."""

from .contracts import (
    CandidateAdapter,
    CandidateSpec,
    ExperimentConfig,
    FoldRequest,
    FoldResult,
)
from .registry import (
    ExperimentProtocolError,
    derive_candidate,
    experiment_registry,
    fold_assignment_sha256,
    load_experiment_config,
    load_fold_assignments,
    registry_payload,
    validate_selection_metric,
)

__all__ = [
    "CandidateAdapter",
    "CandidateSpec",
    "ExperimentConfig",
    "ExperimentProtocolError",
    "FoldRequest",
    "FoldResult",
    "derive_candidate",
    "experiment_registry",
    "fold_assignment_sha256",
    "load_experiment_config",
    "load_fold_assignments",
    "registry_payload",
    "validate_selection_metric",
]
