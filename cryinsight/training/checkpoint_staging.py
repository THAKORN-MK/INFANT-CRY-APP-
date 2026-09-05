"""Native-filesystem staging for repeatedly overwritten Keras checkpoints."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import TracebackType
from typing import Any
import uuid


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        return "microsoft" in os.uname().release.lower()
    except AttributeError:
        return False


def _default_staging_root() -> Path:
    configured = os.environ.get("CRYINSIGHT_CHECKPOINT_STAGING_DIR")
    candidate = Path(configured) if configured else Path(tempfile.gettempdir())
    candidate = candidate.expanduser().resolve()
    if _is_wsl() and (
        candidate.as_posix() == "/mnt" or candidate.as_posix().startswith("/mnt/")
    ):
        candidate = Path("/tmp")
    return candidate / "cryinsight_checkpoints"


def _validate_component(value: str, *, name: str) -> str:
    component = str(value)
    if not component or not _SAFE_COMPONENT.fullmatch(component):
        raise ValueError(
            f"{name} must contain only letters, digits, dot, underscore, or hyphen"
        )
    return component


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointStaging:
    """Stage mutable checkpoints locally, then publish one immutable copy."""

    def __init__(
        self,
        run_id: str,
        fold_name: str,
        filename: str,
        *,
        staging_root: str | Path | None = None,
    ) -> None:
        self.run_id = _validate_component(run_id, name="run_id")
        self.fold_name = _validate_component(fold_name, name="fold_name")
        self.filename = _validate_component(filename, name="filename")
        self.staging_root = (
            Path(staging_root).expanduser().resolve()
            if staging_root is not None
            else _default_staging_root()
        )
        if _is_wsl() and (
            self.staging_root.as_posix() == "/mnt"
            or self.staging_root.as_posix().startswith("/mnt/")
        ):
            raise ValueError("Checkpoint staging must use the native Linux filesystem")
        self._directory: Path | None = None
        self._local_path: Path | None = None

    @property
    def directory(self) -> Path:
        if self._directory is None:
            raise RuntimeError("CheckpointStaging must be entered before use")
        return self._directory

    @property
    def local_path(self) -> Path:
        if self._local_path is None:
            raise RuntimeError("CheckpointStaging must be entered before use")
        return self._local_path

    def __enter__(self) -> "CheckpointStaging":
        if self._directory is not None:
            raise RuntimeError("CheckpointStaging cannot be entered twice")
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self._directory = Path(
            tempfile.mkdtemp(
                prefix=f"{self.run_id}-{self.fold_name}-",
                dir=self.staging_root,
            )
        )
        self._local_path = self._directory / self.filename
        return self

    def publish(self, destination: str | Path) -> dict[str, Any]:
        source = self.local_path
        if not source.is_file():
            raise RuntimeError(f"No selected checkpoint was created at {source}")
        source_size = source.stat().st_size
        if source_size <= 0:
            raise RuntimeError(f"The selected checkpoint is empty: {source}")

        output = Path(destination).resolve()
        if output.exists():
            raise FileExistsError(f"Immutable checkpoint already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        publishing = output.with_name(
            f".{output.name}.{uuid.uuid4().hex}.publishing"
        )
        source_hash = _sha256(source)
        try:
            shutil.copyfile(source, publishing)
            copied_size = publishing.stat().st_size
            copied_hash = _sha256(publishing)
            if copied_size != source_size or copied_hash != source_hash:
                raise OSError(
                    "Checkpoint publication verification failed before final rename"
                )
            os.replace(publishing, output)
        finally:
            if publishing.exists():
                publishing.unlink()

        if output.stat().st_size != source_size or _sha256(output) != source_hash:
            raise OSError("Published checkpoint identity does not match staged checkpoint")
        return {
            "publication_mode": "verified_copy_once",
            "size_bytes": source_size,
            "sha256": source_hash,
        }

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
