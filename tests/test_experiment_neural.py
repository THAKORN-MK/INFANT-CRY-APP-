from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from cryinsight.training.protocol import OriginalRecord


LABELS = ("belly_pain", "burping", "discomfort", "hungry", "tired")


class NeuralAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tensorflow as tf

        cls.tf = tf

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.tf.keras.backend.clear_session()
        self.temporary.cleanup()

    def _spec(self, candidate_id: str):
        from cryinsight.experiments.registry import experiment_registry

        return experiment_registry()[candidate_id]

    def _record(self, label: str, index: int, partition: str) -> OriginalRecord:
        return OriginalRecord(
            record_id=f"{partition}_{label}_{index}",
            filepath=self.root / f"{partition}_{label}_{index}.wav",
            relative_path=f"{label}/{partition}_{index}.wav",
            label=label,
            source_label=label,
            source_dataset="fixture",
            group_id=f"group_{partition}_{label}_{index}",
            group_rule="fixture",
            sha256=f"{index % 10:x}" * 64,
        )

    def _request(self):
        from cryinsight.experiments.contracts import FoldRequest

        train_records = tuple(
            self._record(label, index, "train")
            for label in LABELS
            for index in range(2)
        )
        validation_records = tuple(
            self._record(label, 9, "validation") for label in LABELS
        )
        return FoldRequest(
            experiment_run_id="pipeline__exp_fixture",
            pipeline_run_id="pipeline",
            candidate=self._spec("stage2_logmel_small_cnn"),
            seed=42,
            fold=1,
            train_records=train_records,
            validation_records=validation_records,
            label_order=LABELS,
            output_dir=self.root / "fold_1",
            runtime={
                "epochs": 1,
                "batch_size": 5,
                "verbose": 0,
                "config_sha256": "a" * 64,
                "assignment_sha256": "b" * 64,
            },
        )

    @staticmethod
    def _tiny_tensor_builder(request):
        from cryinsight.experiments.neural import NeuralFoldTensors

        rng = np.random.default_rng(100 + request.fold)
        train_x = rng.normal(size=(len(request.train_records), 16, 16, 1)).astype(
            np.float32
        )
        validation_x = rng.normal(
            size=(len(request.validation_records), 16, 16, 1)
        ).astype(np.float32)
        return NeuralFoldTensors(
            train_features=train_x,
            train_labels=tuple(row.label for row in request.train_records),
            train_sample_ids=tuple(row.record_id for row in request.train_records),
            train_sample_kinds=("original",) * len(request.train_records),
            validation_features=validation_x,
            validation_labels=tuple(row.label for row in request.validation_records),
            validation_record_ids=tuple(
                row.record_id for row in request.validation_records
            ),
            validation_sample_kinds=("original",) * len(request.validation_records),
        )

    def test_factory_maps_corrected_and_multi_branch_models(self) -> None:
        from cryinsight.experiments.neural import build_neural_model

        single = build_neural_model(
            self._spec("stage2_corrected_attention"),
            (196, 128, 1),
            5,
            self.tf,
        )
        multi = build_neural_model(
            self._spec("stage2_multi_branch_attention"),
            (196, 128, 1),
            5,
            self.tf,
        )

        self.assertIn("corrected_single_branch", single.name)
        self.assertIn("corrected_multi_branch", multi.name)

    def test_multi_branch_feature_ablations_remove_missing_branches(self):
        from cryinsight.experiments.neural import build_neural_model
        from cryinsight.experiments.registry import derive_candidate
        for blocks, bins, absent in ((('mfcc', 'delta', 'delta2', 'log_mel'), 184, 'chroma'), (('mfcc', 'delta', 'delta2', 'chroma'), 132, 'mel'), (('mfcc', 'delta', 'delta2'), 120, 'mel')):
            with self.subTest(blocks=blocks):
                candidate = derive_candidate(self._spec('stage2_multi_branch_attention'), 'without_block', {'feature_view': 'feature_block_subset', 'parameters': {'blocks': list(blocks)}})
                model = build_neural_model(candidate, (bins, 16, 1), 2, self.tf)
                values = model(np.zeros((1, bins, 16, 1), dtype=np.float32), training=False)
                self.assertEqual(values.shape, (1, 2))
                self.assertFalse(any(layer.name.startswith(absent + '_') for layer in model.layers))

    def test_normalizer_excludes_augmented_rows_and_binds_manifests(self):
        from cryinsight.experiments.neural import NeuralAdapter
        from cryinsight.training.artefacts import sha256_file
        request = self._request()
        def builder(req):
            base = self._tiny_tensor_builder(req)
            source = req.train_records[0]
            return replace(base, train_features=np.concatenate([base.train_features, np.full((1, 16, 16, 1), 1e5, dtype=np.float32)]), train_labels=(*base.train_labels, source.label), train_sample_ids=(*base.train_sample_ids, 'aug-1'), train_sample_kinds=(*base.train_sample_kinds, 'augmented'), augmentation_manifest=({'sample_id': 'aug-1', 'source_record_id': source.record_id, 'group_id': source.group_id, 'type': 'noise', 'parameters': {'scale': .01}, 'seed': 42},))
        result = NeuralAdapter(tensor_builder=builder, tf_module=self.tf).fit_predict_fold(request)
        manifest = json.loads(result.manifest_path.read_text())
        stats = np.load(result.model_path.parent / 'normalizer.npy')
        self.assertLess(float(np.max(np.abs(stats))), 10)
        for key, name in (('normalizer_manifest_sha256', 'normalizer_manifest.json'), ('augmentation_manifest_sha256', 'augmentation_manifest.json')):
            self.assertEqual(manifest[key], sha256_file(result.model_path.parent / name))
        provenance = json.loads((result.model_path.parent / 'augmentation_manifest.json').read_text())
        self.assertEqual(provenance['waveform_rows'][0]['source_record_id'], request.train_records[0].record_id)


    def test_effective_number_weights_are_finite_and_mean_one(self) -> None:
        from cryinsight.experiments.neural import effective_number_class_weights

        weights = effective_number_class_weights({"a": 10, "b": 100}, beta=0.999)

        self.assertAlmostEqual(sum(weights.values()) / 2.0, 1.0, places=6)
        self.assertGreater(weights["a"], weights["b"])
        self.assertTrue(all(np.isfinite(value) for value in weights.values()))

    def test_focal_loss_bundle_uses_alpha_inside_loss_not_class_weight(self) -> None:
        from cryinsight.experiments.neural import build_loss
        from cryinsight.experiments.registry import derive_candidate

        candidate = derive_candidate(
            self._spec("stage2_corrected_attention"),
            "focal_gamma_2",
            {"loss": "focal", "parameters": {"focal_gamma": 2.0}},
        )
        bundle = build_loss(
            candidate,
            {label: 2 for label in LABELS},
            LABELS,
            self.tf,
        )
        value = bundle.loss(
            self.tf.one_hot([0], depth=5),
            self.tf.constant([[0.8, 0.05, 0.05, 0.05, 0.05]], dtype=self.tf.float32),
        )

        self.assertIsNone(bundle.class_weight)
        self.assertTrue(np.isfinite(np.asarray(value)).all())
        self.assertEqual(bundle.metadata["gamma"], 2.0)

    def test_tiny_neural_fold_publishes_loadable_keras_checkpoint(self) -> None:
        from cryinsight.experiments.neural import NeuralAdapter

        adapter = NeuralAdapter(
            tensor_builder=self._tiny_tensor_builder,
            tf_module=self.tf,
        )
        result = adapter.fit_predict_fold(self._request())
        loaded = self.tf.keras.models.load_model(result.model_path)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded.output_shape[-1], 5)
        self.assertEqual(result.probabilities.shape, (5, 5))
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-5)
        self.assertGreater(manifest["parameter_count"], 0)
        self.assertEqual(
            manifest["normalizer_fit_record_ids"],
            [row.record_id for row in self._request().train_records],
        )
        self.assertTrue((result.model_path.parent / "checkpoint_publication.json").is_file())


if __name__ == "__main__":
    unittest.main()
