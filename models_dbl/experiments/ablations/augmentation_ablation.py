"""Stage 2 augmentation and Mixup ablations."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "cryinsight").is_dir():
        sys.path.insert(0, str(ancestor))
        break

from Models_dbl.experiments.script_support import audit_definition_cli, build_definition

EXPERIMENT_ID = "stage2_augmentation_mixup"


def experiment_variants() -> dict[str, dict[str, bool]]:
    return {
        "none": {"waveform_augmentation": False, "mixup": False},
        "waveform_only": {"waveform_augmentation": True, "mixup": False},
        "waveform_plus_mixup": {"waveform_augmentation": True, "mixup": True},
    }


def build_augmentation_plan(variant: str) -> dict[str, bool]:
    variants = experiment_variants()
    if variant not in variants:
        raise ValueError(f"Unknown augmentation ablation variant: {variant}")
    return dict(variants[variant])


def experiment_definition():
    return build_definition(
        EXPERIMENT_ID,
        factory="build_augmentation_plan",
        purpose="Measure waveform augmentation and Mixup contributions separately.",
        input_contract="Augmentation is generated from training-fold originals only; validation and held-out Test remain original-only.",
        variants=experiment_variants(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    return audit_definition_cli(experiment_definition(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
