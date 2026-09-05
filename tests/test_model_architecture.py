from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow is unavailable")
class CorrectedArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tensorflow as tf

        cls.tf = tf

    def tearDown(self) -> None:
        self.tf.keras.backend.clear_session()

    def test_stage1_bilstm_sequence_is_32_time_steps(self) -> None:
        from cryinsight.models.stage1_model import build_stage1_model

        model = build_stage1_model(self.tf, (120, 128, 1), 2)
        self.assertEqual(tuple(model.get_layer("cnn_block_3_pool").output.shape[1:3]), (15, 32))
        self.assertEqual(model.get_layer("time_sequence").output.shape[1], 32)
        self.assertEqual(model.get_layer("classifier").dtype, "float32")

    def test_stage2_single_branch_keeps_32_time_steps(self) -> None:
        from cryinsight.models.stage2_model import build_stage2_model

        model = build_stage2_model(
            self.tf, (196, 128, 1), 5, architecture="corrected_single_branch"
        )
        self.assertEqual(tuple(model.get_layer("cnn_block_4_pool").output.shape[1:3]), (12, 32))
        self.assertEqual(model.get_layer("time_sequence").output.shape[1], 32)

    def test_stage2_multi_branch_uses_declared_feature_boundaries(self) -> None:
        from cryinsight.models.stage2_model import FEATURE_BLOCKS, build_stage2_model

        self.assertEqual(
            FEATURE_BLOCKS,
            {"mfcc_derivatives": (0, 120), "log_mel": (120, 184), "chroma": (184, 196)},
        )
        model = build_stage2_model(
            self.tf, (196, 128, 1), 5, architecture="corrected_multi_branch"
        )
        self.assertEqual(model.get_layer("time_sequence").output.shape[1], 32)

    def test_attention_model_round_trips_in_keras_format(self) -> None:
        from cryinsight.models.stage1_model import build_stage1_model

        model = build_stage1_model(self.tf, (120, 128, 1), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            loaded = self.tf.keras.models.load_model(path)
        self.assertEqual(loaded.output_shape, (None, 2))


if __name__ == "__main__":
    unittest.main()
