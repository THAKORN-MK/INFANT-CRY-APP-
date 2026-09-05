"""Stage 2 feature-block removal ablations."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "cryinsight").is_dir():
        sys.path.insert(0, str(ancestor))
        break

from Models_dbl.experiments.script_support import audit_definition_cli, build_definition

EXPERIMENT_ID = "stage2_feature_blocks"
ALL_FEATURES = ("mfcc", "delta", "delta2", "log_mel", "chroma")


def experiment_variants() -> dict[str, dict[str, object]]:
    variants = {"all_features": {"features": list(ALL_FEATURES)}}
    for removed in ("delta", "delta2", "log_mel", "chroma"):
        variants[f"without_{removed}"] = {
            "features": [feature for feature in ALL_FEATURES if feature != removed],
            "removed": removed,
        }
    return variants


def build_feature_plan(variant: str) -> tuple[str, ...]:
    variants = experiment_variants()
    if variant not in variants:
        raise ValueError(f"Unknown feature ablation variant: {variant}")
    return tuple(variants[variant]["features"])


def experiment_definition():
    return build_definition(
        EXPERIMENT_ID,
        factory="build_feature_plan",
        purpose="Measure the contribution of Delta, Delta2, Log-Mel, and Chroma feature blocks.",
        input_contract="Each variant uses the same records and grouped folds; normalization is refitted on the retained training-fold blocks.",
        variants=experiment_variants(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    return audit_definition_cli(experiment_definition(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
