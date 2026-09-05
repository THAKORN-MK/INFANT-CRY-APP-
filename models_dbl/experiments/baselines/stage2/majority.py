"""Most-frequent-class Stage 2 baseline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "cryinsight").is_dir():
        sys.path.insert(0, str(ancestor))
        break

from Models_dbl.experiments.script_support import audit_definition_cli, build_definition

EXPERIMENT_ID = "stage2_majority"


def build_estimator(seed: int = 42):
    from cryinsight.experiments.classical import build_classical_estimator
    from cryinsight.experiments.registry import experiment_registry

    return build_classical_estimator(experiment_registry()[EXPERIMENT_ID], seed)


def experiment_variants() -> dict[str, dict[str, str]]:
    return {"most_frequent": {"strategy": "most_frequent"}}


def experiment_definition():
    return build_definition(
        EXPERIMENT_ID,
        factory="build_estimator",
        purpose="Lower-bound five-class classifier using only the training-fold class distribution.",
        input_contract="Training-fold labels only; no held-out Test access.",
        variants=experiment_variants(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    return audit_definition_cli(experiment_definition(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
