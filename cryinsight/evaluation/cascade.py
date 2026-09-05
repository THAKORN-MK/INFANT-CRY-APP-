"""End-to-end Stage 1 gate plus Stage 2 route evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


def aggregate_cascade_rows(
    stage1_rows: Sequence[Mapping[str, str]],
    stage2_rows: Sequence[Mapping[str, str]],
) -> dict[str, float | int | str]:
    stage2_by_id = {row["record_id"]: row for row in stage2_rows}
    if len(stage2_by_id) != len(stage2_rows):
        raise ValueError("Stage 2 predictions contain duplicate record IDs")
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    false_rejects = 0
    missing_routed = 0
    for gate in stage1_rows:
        record_id = gate["record_id"]
        if gate["label"] == "not_baby":
            true_label = "not_baby"
        else:
            detail = stage2_by_id.get(record_id)
            if detail is None:
                raise ValueError(f"Missing Stage 2 truth/prediction for infant {record_id}")
            true_label = detail["label"]

        if gate["predicted_label"] != "baby":
            predicted_label = "not_baby"
            if gate["label"] == "baby":
                false_rejects += 1
        else:
            detail = stage2_by_id.get(record_id)
            if detail is None:
                predicted_label = "stage2_missing"
                missing_routed += 1
            else:
                predicted_label = detail["predicted_label"]
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
    correct = sum(
        true == predicted for true, predicted in zip(true_labels, predicted_labels)
    )
    support = len(true_labels)
    if support == 0:
        raise ValueError("Cascade evaluation support is empty")
    return {
        "evaluation_scope": "end_to_end_cascade_on_supplied_prediction_support",
        "support": support,
        "correct": correct,
        "accuracy": correct / support,
        "stage1_false_rejects": false_rejects,
        "stage2_missing_after_positive_gate": missing_routed,
    }


def read_rows(path: str | Path) -> tuple[dict[str, str], ...]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def evaluate_prediction_files(
    stage1_predictions: str | Path,
    stage2_predictions: str | Path,
) -> dict[str, float | int | str]:
    return aggregate_cascade_rows(
        read_rows(stage1_predictions), read_rows(stage2_predictions)
    )
