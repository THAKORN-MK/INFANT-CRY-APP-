"""Resolve an immutable Stage 1 run for a paired Stage 2 run."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


class RunPairingError(RuntimeError):
    """Raised when Stage 2 cannot safely pair with a Stage 1 run."""


@dataclass(frozen=True)
class StageRunIdentity:
    run_id: str
    run_dir: Path
    created_at: str
    verification_status: str
    verification_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "created_at": self.created_at,
            "verification_status": self.verification_status,
            "verification_sha256": self.verification_sha256,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunPairingError(f"Cannot read valid run metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise RunPairingError(f"Run metadata must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_key(run_dir: Path) -> tuple[str, str]:
    protocol_path = run_dir / "protocol.json"
    protocol = _read_json(protocol_path)
    protocol_run_id = str(protocol.get("run_id", ""))
    if protocol_run_id and protocol_run_id != run_dir.name:
        raise RunPairingError(
            f"Protocol run_id {protocol_run_id!r} does not match directory {run_dir.name!r}"
        )
    return str(protocol.get("created_at", "")), run_dir.name


def _validate_complete_run(run_dir: Path) -> StageRunIdentity:
    protocol = _read_json(run_dir / "protocol.json")
    verification_path = run_dir / "verification.json"
    if not verification_path.is_file():
        raise RunPairingError(
            f"Latest Stage 1 run {run_dir.name} is still training; "
            "verification.json is not available yet"
        )
    verification = _read_json(verification_path)
    verification_run_id = str(verification.get("run_id", ""))
    if verification_run_id != run_dir.name:
        raise RunPairingError(
            f"Verification run_id {verification_run_id!r} does not match "
            f"directory {run_dir.name!r}"
        )
    status = str(verification.get("status", ""))
    if status != "complete":
        raise RunPairingError(
            f"Stage 1 run {run_dir.name} is not complete (status={status!r})"
        )
    return StageRunIdentity(
        run_id=run_dir.name,
        run_dir=run_dir.resolve(),
        created_at=str(protocol.get("created_at", "")),
        verification_status=status,
        verification_sha256=_sha256(verification_path),
    )


def resolve_stage1_run(
    binary_stage_root: str | Path,
    *,
    requested_run_id: str | None = None,
) -> StageRunIdentity:
    """Return the requested or newest Stage 1 run, only when it is complete.

    Automatic selection intentionally validates the newest run and never falls
    back to an older completed run.  This prevents Stage 2 from silently pairing
    with stale Stage 1 artefacts while a new Stage 1 run is still in progress.
    """

    runs_dir = Path(binary_stage_root).resolve() / "runs"
    if requested_run_id:
        run_dir = runs_dir / requested_run_id
        if not run_dir.is_dir():
            raise RunPairingError(f"Stage 1 run does not exist: {run_dir}")
        return _validate_complete_run(run_dir)

    if not runs_dir.is_dir():
        raise RunPairingError(f"Stage 1 runs directory does not exist: {runs_dir}")
    candidates = tuple(path for path in runs_dir.iterdir() if path.is_dir())
    if not candidates:
        raise RunPairingError(f"No Stage 1 runs were found in: {runs_dir}")
    newest = max(candidates, key=_candidate_key)
    return _validate_complete_run(newest)
