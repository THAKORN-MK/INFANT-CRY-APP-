from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path


class CheckpointStagingTests(unittest.TestCase):
    def test_publish_copies_checkpoint_once_and_verifies_identity(self) -> None:
        from cryinsight.training.checkpoint_staging import CheckpointStaging

        payload = b"selected keras checkpoint"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "published" / "fold_1.keras"
            with CheckpointStaging(
                "run_1",
                "fold_1",
                "fold_1.keras",
                staging_root=root / "native",
            ) as staging:
                staging.local_path.write_bytes(payload)
                staging_directory = staging.directory
                manifest = staging.publish(destination)

                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(manifest["size_bytes"], len(payload))
                self.assertEqual(
                    manifest["sha256"],
                    hashlib.sha256(payload).hexdigest(),
                )
                self.assertEqual(manifest["publication_mode"], "verified_copy_once")

            self.assertFalse(staging_directory.exists())

    def test_publish_refuses_to_overwrite_an_immutable_destination(self) -> None:
        from cryinsight.training.checkpoint_staging import CheckpointStaging

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "fold_1.keras"
            destination.write_bytes(b"existing")
            with CheckpointStaging(
                "run_1",
                "fold_1",
                "fold_1.keras",
                staging_root=root / "native",
            ) as staging:
                staging.local_path.write_bytes(b"replacement")
                with self.assertRaises(FileExistsError):
                    staging.publish(destination)

            self.assertEqual(destination.read_bytes(), b"existing")

    def test_publish_requires_a_nonempty_selected_checkpoint(self) -> None:
        from cryinsight.training.checkpoint_staging import CheckpointStaging

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with CheckpointStaging(
                "run_1",
                "fold_1",
                "fold_1.keras",
                staging_root=root / "native",
            ) as staging:
                with self.assertRaisesRegex(RuntimeError, "selected checkpoint"):
                    staging.publish(root / "published.keras")

                staging.local_path.touch()
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    staging.publish(root / "published.keras")

    @unittest.skipUnless(
        os.name == "posix" and "microsoft" in os.uname().release.lower(),
        "WSL-specific native-filesystem contract",
    )
    def test_default_wsl_staging_directory_is_not_under_mnt(self) -> None:
        from cryinsight.training.checkpoint_staging import CheckpointStaging

        with CheckpointStaging("run_1", "fold_1", "fold_1.keras") as staging:
            self.assertFalse(str(staging.directory).startswith("/mnt/"))


if __name__ == "__main__":
    unittest.main()
