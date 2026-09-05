from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "Models_dbl" / "Main" / "train_main_dbl.py"


def load_main_module():
    spec = importlib.util.spec_from_file_location("stage2_pairing_integration", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageRunPairingIntegrationTests(unittest.TestCase):
    def test_prepare_only_reuses_completed_binary_run_id_and_records_pairing(self) -> None:
        module = load_main_module()
        run_id = "20260821T010000Z_pair"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_dir = root / "split" / "train"
            test_dir = root / "split" / "test"
            main_root = root / "Models_dbl" / "Main"
            binary_root = root / "Models_dbl" / "binary"
            binary_run = binary_root / "runs" / run_id
            binary_run.mkdir(parents=True)
            (binary_run / "protocol.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "created_at": "2026-08-21T01:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            (binary_run / "verification.json").write_text(
                json.dumps({"run_id": run_id, "status": "complete"}),
                encoding="utf-8",
            )

            for label in module.LABEL_ORDER:
                for index in range(15):
                    path = train_dir / label / f"train_{index}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"train-{label}-{index}".encode())
                heldout = test_dir / label / "test.wav"
                heldout.parent.mkdir(parents=True, exist_ok=True)
                heldout.write_bytes(f"test-{label}".encode())

            with redirect_stdout(io.StringIO()):
                exit_code = module.main(
                    [
                        "--prepare-only",
                        "--train-data-dir",
                        str(train_dir),
                        "--test-data-dir",
                        str(test_dir),
                        "--stage-root",
                        str(main_root),
                        "--binary-stage-root",
                        str(binary_root),
                    ]
                )

            self.assertEqual(exit_code, 0)
            prepared_run = main_root / "runs" / run_id
            self.assertTrue(prepared_run.is_dir())
            protocol = json.loads(
                (prepared_run / "protocol.json").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol["run_id"], run_id)
            self.assertEqual(protocol["paired_stage1_run"]["run_id"], run_id)
            self.assertEqual(
                protocol["paired_stage1_run"]["verification_status"],
                "complete",
            )
            self.assertEqual(
                len(protocol["paired_stage1_run"]["verification_sha256"]),
                64,
            )


if __name__ == "__main__":
    unittest.main()
