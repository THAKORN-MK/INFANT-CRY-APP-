from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class StageRunPairingTests(unittest.TestCase):
    def _write_run(
        self,
        runs_dir: Path,
        run_id: str,
        *,
        created_at: str,
        status: str | None,
    ) -> Path:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "protocol.json").write_text(
            json.dumps({"run_id": run_id, "created_at": created_at}),
            encoding="utf-8",
        )
        if status is not None:
            (run_dir / "verification.json").write_text(
                json.dumps({"run_id": run_id, "status": status}),
                encoding="utf-8",
            )
        return run_dir

    def test_automatic_pairing_selects_newest_complete_stage1_run(self) -> None:
        from cryinsight.training.run_pairing import resolve_stage1_run

        with tempfile.TemporaryDirectory() as directory:
            stage_root = Path(directory) / "binary"
            runs_dir = stage_root / "runs"
            self._write_run(
                runs_dir,
                "20260820T010000Z_old",
                created_at="2026-08-20T01:00:00+00:00",
                status="complete",
            )
            newest = self._write_run(
                runs_dir,
                "20260821T010000Z_new",
                created_at="2026-08-21T01:00:00+00:00",
                status="complete",
            )

            identity = resolve_stage1_run(stage_root)

            self.assertEqual(identity.run_id, "20260821T010000Z_new")
            self.assertEqual(identity.run_dir, newest.resolve())
            self.assertEqual(identity.verification_status, "complete")
            self.assertEqual(len(identity.verification_sha256), 64)

    def test_automatic_pairing_refuses_newest_run_while_training(self) -> None:
        from cryinsight.training.run_pairing import RunPairingError, resolve_stage1_run

        with tempfile.TemporaryDirectory() as directory:
            stage_root = Path(directory) / "binary"
            runs_dir = stage_root / "runs"
            self._write_run(
                runs_dir,
                "20260820T010000Z_complete",
                created_at="2026-08-20T01:00:00+00:00",
                status="complete",
            )
            self._write_run(
                runs_dir,
                "20260821T010000Z_training",
                created_at="2026-08-21T01:00:00+00:00",
                status=None,
            )

            with self.assertRaisesRegex(RunPairingError, "still training"):
                resolve_stage1_run(stage_root)

    def test_explicit_pairing_refuses_incomplete_run(self) -> None:
        from cryinsight.training.run_pairing import RunPairingError, resolve_stage1_run

        with tempfile.TemporaryDirectory() as directory:
            stage_root = Path(directory) / "binary"
            self._write_run(
                stage_root / "runs",
                "20260821T010000Z_failed",
                created_at="2026-08-21T01:00:00+00:00",
                status="incomplete",
            )

            with self.assertRaisesRegex(RunPairingError, "not complete"):
                resolve_stage1_run(
                    stage_root,
                    requested_run_id="20260821T010000Z_failed",
                )

    def test_pairing_rejects_verification_run_id_mismatch(self) -> None:
        from cryinsight.training.run_pairing import RunPairingError, resolve_stage1_run

        with tempfile.TemporaryDirectory() as directory:
            stage_root = Path(directory) / "binary"
            run_dir = self._write_run(
                stage_root / "runs",
                "20260821T010000Z_expected",
                created_at="2026-08-21T01:00:00+00:00",
                status="complete",
            )
            (run_dir / "verification.json").write_text(
                json.dumps({"run_id": "different", "status": "complete"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RunPairingError, "does not match"):
                resolve_stage1_run(stage_root)


if __name__ == "__main__":
    unittest.main()
