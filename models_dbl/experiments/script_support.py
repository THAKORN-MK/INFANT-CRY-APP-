"""Side-effect-free CLI support shared by experiment definition scripts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from typing import Any, Sequence

from cryinsight.training.experiments import experiment_registry


def build_definition(
    experiment_id: str,
    *,
    factory: str,
    purpose: str,
    variants: dict[str, Any],
    input_contract: str,
) -> dict[str, Any]:
    registry = experiment_registry()
    if experiment_id not in registry:
        raise ValueError(f"Unknown experiment ID: {experiment_id}")
    return {
        "experiment_id": experiment_id,
        "registry": asdict(registry[experiment_id]),
        "implementation": {
            "factory": factory,
            "purpose": purpose,
            "input_contract": input_contract,
            "variants": variants,
        },
    }


def audit_definition_cli(
    definition: dict[str, Any],
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=f"Audit experiment definition {definition['experiment_id']}"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        required=True,
        help="Print the registered definition without training or writing artefacts.",
    )
    parser.parse_args(argv)
    print(
        json.dumps(
            {
                **definition,
                "status": "definition_ready",
                "training_started": False,
                "metrics": "not_run",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
