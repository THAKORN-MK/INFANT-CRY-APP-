"""Compatibility exports for the shared experiment package."""

from cryinsight.experiments.contracts import CandidateSpec as ExperimentSpec
from cryinsight.experiments.registry import (
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
    "ExperimentProtocolError",
    "ExperimentSpec",
    "derive_candidate",
    "experiment_registry",
    "fold_assignment_sha256",
    "load_experiment_config",
    "load_fold_assignments",
    "registry_payload",
    "validate_selection_metric",
]
