"""Frozen YAMNet embedding baselines with linear and MLP heads."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "cryinsight").is_dir():
        PROJECT_ROOT = ancestor
        sys.path.insert(0, str(ancestor))
        break
else:
    raise RuntimeError("Could not locate project root")

from Models_dbl.experiments.script_support import audit_definition_cli, build_definition

EXPERIMENT_ID = "stage2_yamnet_linear"
YAMNET_ARCHIVE = PROJECT_ROOT / "yamnet-tensorflow2-yamnet-v1.tar.gz"


def build_estimator(*, head: str = "linear", seed: int = 42):
    from cryinsight.experiments.classical import build_classical_estimator
    from cryinsight.experiments.registry import experiment_registry

    candidate_id = {
        "linear": "stage2_yamnet_linear",
        "mlp": "stage2_yamnet_mlp",
    }.get(head)
    if candidate_id is None:
        raise ValueError("head must be 'linear' or 'mlp'")
    return build_classical_estimator(experiment_registry()[candidate_id], seed)


def experiment_variants() -> dict[str, dict[str, object]]:
    return {
        "linear": {"registry_id": "stage2_yamnet_linear", "head": "linear"},
        "mlp": {"registry_id": "stage2_yamnet_mlp", "head": "mlp", "hidden_units": 256},
    }


def experiment_definition():
    payload = build_definition(
        EXPERIMENT_ID,
        factory="build_estimator",
        purpose="Compare frozen YAMNet embeddings with linear and one-hidden-layer heads.",
        input_contract="Per-clip YAMNet embeddings extracted without fine-tuning; scaler/head fitted within each training fold.",
        variants=experiment_variants(),
    )
    payload["implementation"]["local_archive"] = str(YAMNET_ARCHIVE)
    payload["implementation"]["local_archive_available"] = YAMNET_ARCHIVE.is_file()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    return audit_definition_cli(experiment_definition(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
