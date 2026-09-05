"""Small Log-Mel CNN Stage 1 baseline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "cryinsight").is_dir():
        sys.path.insert(0, str(ancestor))
        break

from Models_dbl.experiments.script_support import audit_definition_cli, build_definition

EXPERIMENT_ID = "stage1_logmel_small_cnn"


def build_model(input_shape: Sequence[int], num_classes: int = 2):
    import tensorflow as tf
    from cryinsight.experiments.neural import build_small_cnn

    return build_small_cnn(tf, input_shape, num_classes)


def experiment_variants() -> dict[str, dict[str, object]]:
    return {"small_cnn": {"conv_filters": [24, 48], "feature": "log_mel"}}


def experiment_definition():
    return build_definition(
        EXPERIMENT_ID,
        factory="build_model",
        purpose="Compact neural baseline without BiLSTM or Attention.",
        input_contract="Training-fold-normalized Log-Mel tensors; augmentation is training-fold only.",
        variants=experiment_variants(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    return audit_definition_cli(experiment_definition(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
