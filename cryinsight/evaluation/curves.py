"""ROC and precision-recall data derived from prediction artefacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cryinsight.training.artefacts import normalize_probability_rows


def compute_roc_pr_tables(
    true_labels: Sequence[str],
    scores: np.ndarray,
    label_order: Sequence[str],
) -> dict[str, dict[str, Any]]:
    try:
        from sklearn import metrics
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for ROC/PR curves") from exc
    labels = tuple(label_order)
    if len(true_labels) != len(scores):
        raise ValueError("Label and probability support differ")
    probabilities = normalize_probability_rows(scores)
    index = {label: position for position, label in enumerate(labels)}
    try:
        true_indices = np.asarray([index[label] for label in true_labels], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unknown true label: {exc.args[0]}") from exc
    tables: dict[str, dict[str, Any]] = {}
    for class_index, label in enumerate(labels):
        binary_true = (true_indices == class_index).astype(int)
        if len(np.unique(binary_true)) < 2:
            raise ValueError(f"ROC/PR for {label} requires positive and negative samples")
        false_positive_rate, true_positive_rate, roc_thresholds = metrics.roc_curve(
            binary_true, probabilities[:, class_index]
        )
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            binary_true, probabilities[:, class_index]
        )
        tables[label] = {
            "roc_auc": float(metrics.auc(false_positive_rate, true_positive_rate)),
            "average_precision": float(
                metrics.average_precision_score(binary_true, probabilities[:, class_index])
            ),
            "roc": {
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "threshold": roc_thresholds,
            },
            "pr": {
                "precision": precision,
                "recall": recall,
                "threshold": pr_thresholds,
            },
        }
    return tables


def read_prediction_csv(
    path: str | Path, label_order: Sequence[str]
) -> tuple[tuple[str, ...], np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    score_fields = tuple(f"score_{label}" for label in label_order)
    labels = tuple(row["label"] for row in rows)
    scores = np.asarray(
        [[float(row[field]) for field in score_fields] for row in rows],
        dtype=np.float64,
    )
    return labels, normalize_probability_rows(scores)


def write_curve_csvs(
    output_dir: str | Path,
    tables: dict[str, dict[str, Any]],
) -> tuple[Path, ...]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for label, table in tables.items():
        roc_path = root / f"roc_{label}.csv"
        pr_path = root / f"pr_{label}.csv"
        if roc_path.exists() or pr_path.exists():
            raise FileExistsError("Refusing to replace immutable curve artefact")
        with roc_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("false_positive_rate", "true_positive_rate", "threshold"))
            writer.writerows(zip(
                table["roc"]["false_positive_rate"],
                table["roc"]["true_positive_rate"],
                table["roc"]["threshold"],
            ))
        thresholds = np.append(table["pr"]["threshold"], np.nan)
        with pr_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("precision", "recall", "threshold"))
            writer.writerows(zip(
                table["pr"]["precision"], table["pr"]["recall"], thresholds
            ))
        outputs.extend((roc_path, pr_path))
    return tuple(outputs)


def write_curve_pngs(
    output_dir: str | Path,
    tables: dict[str, dict[str, Any]],
    *,
    title_prefix: str,
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for ROC/PR plots") from exc
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    roc_path = root / "roc_curves.png"
    pr_path = root / "precision_recall_curves.png"
    if roc_path.exists() or pr_path.exists():
        raise FileExistsError("Refusing to replace immutable curve plot")

    fig, axis = plt.subplots(figsize=(7, 6))
    for label, table in tables.items():
        axis.plot(
            table["roc"]["false_positive_rate"],
            table["roc"]["true_positive_rate"],
            label=f"{label} (AUC={table['roc_auc']:.3f})",
        )
    axis.plot((0, 1), (0, 1), linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{title_prefix} ROC")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    for label, table in tables.items():
        axis.plot(
            table["pr"]["recall"],
            table["pr"]["precision"],
            label=f"{label} (AP={table['average_precision']:.3f})",
        )
    axis.set(xlabel="Recall", ylabel="Precision", title=f"{title_prefix} precision-recall")
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=180)
    plt.close(fig)
    return roc_path, pr_path
