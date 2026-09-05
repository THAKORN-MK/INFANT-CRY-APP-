"""Stage 2 CNN + BiLSTM without Attention ablation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "cryinsight").is_dir():
        sys.path.insert(0, str(ancestor))
        break

from Models_dbl.experiments.script_support import audit_definition_cli, build_definition

EXPERIMENT_ID = "stage2_cnn_bilstm"


def build_model(input_shape: Sequence[int], num_classes: int = 5):
    import tensorflow as tf
    from cryinsight.experiments.neural import build_cnn_bilstm

    return build_cnn_bilstm(tf, input_shape, num_classes)


def experiment_variants() -> dict[str, dict[str, object]]:
    return {"cnn_bilstm": {"bilstm": True, "attention": False}}


def experiment_definition():
    return build_definition(
        EXPERIMENT_ID,
        factory="build_model",
        purpose="Isolate the contribution of Attention after CNN + BiLSTM.",
        input_contract="Same Stage 2 features, folds, normalization, augmentation, and selection metric as the proposed model.",
        variants=experiment_variants(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    return audit_definition_cli(experiment_definition(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
