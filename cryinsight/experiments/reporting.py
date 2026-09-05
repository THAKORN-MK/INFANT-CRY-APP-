"""Deterministic, write-once reports for verified grouped-OOF experiments."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cryinsight.training.protocol import _write_text_once


LEADERBOARD_COLUMNS = (
    "rank",
    "candidate_id",
    "wave",
    "seeds",
    "oof_macro_f1_mean",
    "oof_macro_f1_std",
    "oof_balanced_accuracy_mean",
    "oof_accuracy_mean",
    "minimum_class_recall_mean",
    "parameter_count",
    "verification_status",
)


def _first(row: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def _leaderboard_row(row: Mapping[str, Any]) -> dict[str, Any]:
    seeds = row.get("seeds", [row["seed"]] if "seed" in row else [])
    if isinstance(seeds, (list, tuple)):
        seed_text = ";".join(str(value) for value in seeds)
    else:
        seed_text = str(seeds)
    return {
        "rank": row.get("rank", ""),
        "candidate_id": row.get("candidate_id", ""),
        "wave": row.get("wave", ""),
        "seeds": seed_text,
        "oof_macro_f1_mean": _first(
            row, "mean_oof_macro_f1", "oof_macro_f1", default=""
        ),
        "oof_macro_f1_std": _first(row, "std_oof_macro_f1", default=0.0),
        "oof_balanced_accuracy_mean": _first(
            row,
            "mean_oof_balanced_accuracy",
            "oof_balanced_accuracy",
            default="",
        ),
        "oof_accuracy_mean": _first(
            row, "mean_oof_accuracy", "oof_accuracy", default=""
        ),
        "minimum_class_recall_mean": _first(
            row,
            "mean_minimum_class_recall",
            "minimum_class_recall",
            default="",
        ),
        "parameter_count": row.get("parameter_count", ""),
        "verification_status": row.get("verification_status", ""),
    }


def _csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEADERBOARD_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_leaderboard_row(row))
    return buffer.getvalue()


def write_leaderboard_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    _write_text_once(Path(path), _csv_text(rows))


def _format_metric(value: Any) -> str:
    if value == "" or value is None:
        return "—"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _leaderboard_markdown(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "## Verified candidate leaderboard",
        "",
        "| Rank | Candidate | Wave | Seeds | OOF Macro F1 | OOF Balanced Accuracy | OOF Accuracy | Minimum class recall | Parameters | Verification |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for raw in rows:
        row = _leaderboard_row(raw)
        lines.append(
            "| {rank} | {candidate_id} | {wave} | {seeds} | {macro} | {balanced} | "
            "{accuracy} | {recall} | {parameters} | {verification} |".format(
                rank=row["rank"],
                candidate_id=row["candidate_id"],
                wave=row["wave"],
                seeds=row["seeds"],
                macro=_format_metric(row["oof_macro_f1_mean"]),
                balanced=_format_metric(row["oof_balanced_accuracy_mean"]),
                accuracy=_format_metric(row["oof_accuracy_mean"]),
                recall=_format_metric(row["minimum_class_recall_mean"]),
                parameters=row["parameter_count"],
                verification=row["verification_status"],
            )
        )
    lines.extend(["", "## Excluded/failed candidates", ""])
    if failures:
        lines.extend(
            f"- `{row.get('candidate_id', 'unknown')}` — {row.get('reason', 'not verified')}"
            for row in failures
        )
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def write_leaderboard_markdown(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]] = (),
) -> None:
    _write_text_once(
        Path(path),
        _leaderboard_markdown(tuple(rows), tuple(failures)),
    )


def write_promotion_recommendation(
    path: str | Path,
    decision: Mapping[str, Any],
) -> None:
    checks = decision.get("checks", {})
    if not isinstance(checks, Mapping):
        checks = {}
    lines = [
        "# Promotion recommendation",
        "",
        f"Status: `{decision.get('status', 'not_available')}`",
        "",
        "The decision uses corrected grouped OOF only; Final Test was unavailable to ranking.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if bool(value) else 'FAIL'} — `{name}`"
        for name, value in checks.items()
    )
    _write_text_once(Path(path), "\n".join(lines) + "\n")


def _remove_test_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if "final_test" in normalized or "heldout" in normalized or "held_out" in normalized:
                continue
            output[str(key)] = _remove_test_evidence(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_remove_test_evidence(item) for item in value]
    return value


def _payload_hash(payload: Mapping[str, Any]) -> str:
    sanitized = _remove_test_evidence(payload)
    encoded = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_experiment_report(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """Write one comparison report without rendering any Final Test metric."""

    ranked = tuple(payload.get("ranked_candidates", ()))
    failures = tuple(payload.get("exclusions", ()))
    reference = _remove_test_evidence(payload.get("reference", {}))
    if not isinstance(reference, Mapping):
        reference = {}
    lines = [
        "# Shared Experiment Comparison",
        "",
        f"Experiment run: `{payload.get('experiment_run_id', 'unknown')}`",
        f"Wave: `{payload.get('wave', 'unknown')}`",
        f"Verified payload SHA-256: `{_payload_hash(payload)}`",
        "",
        "Evaluation and ranking scope: **corrected grouped OOF only**.",
        "Final Test was unavailable to ranking, model selection, and promotion checks.",
        "",
        "## Reference OOF metrics",
        "",
    ]
    for key in (
        "oof_macro_f1",
        "oof_balanced_accuracy",
        "oof_accuracy",
        "minimum_class_recall",
    ):
        if key in reference:
            lines.append(f"- `{key}`: {_format_metric(reference[key])}")
    lines.extend(["", _leaderboard_markdown(ranked, failures).rstrip(), ""])

    per_class = payload.get("per_class_oof", {})
    lines.extend(["## Per-class OOF comparison", ""])
    if isinstance(per_class, Mapping) and per_class:
        for candidate_id, metrics in per_class.items():
            if isinstance(metrics, Mapping):
                values = ", ".join(
                    f"{label}={_format_metric(value)}"
                    for label, value in sorted(metrics.items())
                )
                lines.append(f"- `{candidate_id}`: {values}")
    else:
        lines.append("- Not available.")

    lines.extend(["", "## Stability across seeds", ""])
    repeated = [row for row in ranked if len(row.get("seeds", ())) > 1]
    if repeated:
        for row in repeated:
            lines.append(
                f"- `{row.get('candidate_id')}`: OOF Macro F1 SD "
                f"{_format_metric(row.get('std_oof_macro_f1'))}"
            )
    else:
        lines.append("- Single-seed screening; stability confirmation is pending.")

    promotion = payload.get("promotion", {})
    lines.extend(["", "## Promotion checks", ""])
    if isinstance(promotion, Mapping):
        lines.append(f"Status: `{promotion.get('status', 'not_available')}`")
        checks = promotion.get("checks", {})
        if isinstance(checks, Mapping):
            lines.extend(
                f"- {'PASS' if bool(value) else 'FAIL'} — `{name}`"
                for name, value in checks.items()
            )

    lines.extend(["", "## Limitations", ""])
    limitations = payload.get("limitations", ())
    if isinstance(limitations, (list, tuple)) and limitations:
        lines.extend(f"- {value}" for value in limitations)
    else:
        lines.append("- Results are internal grouped cross-validation estimates.")
    _write_text_once(Path(path), "\n".join(lines) + "\n")


__all__ = [
    "LEADERBOARD_COLUMNS",
    "write_experiment_report",
    "write_leaderboard_csv",
    "write_leaderboard_markdown",
    "write_promotion_recommendation",
]
