from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest


class ExperimentContractTests(unittest.TestCase):
    def test_candidate_identity_is_stable_and_top_level_config_is_immutable(self) -> None:
        from cryinsight.experiments.contracts import CandidateSpec

        candidate = CandidateSpec(
            candidate_id="stage2_majority",
            stage="stage2",
            family="baseline",
            feature_view="labels_only",
            adapter="classical",
            model="dummy_most_frequent",
            augmentation="none",
            normalization="none",
            loss="not_applicable",
        )

        self.assertEqual(candidate.selection_metric, "oof_macro_f1")
        with self.assertRaises(FrozenInstanceError):
            candidate.model = "changed"  # type: ignore[misc]

    def test_candidate_rejects_unknown_stage_and_adapter(self) -> None:
        from cryinsight.experiments.contracts import CandidateSpec

        common = {
            "candidate_id": "invalid",
            "family": "baseline",
            "feature_view": "labels_only",
            "model": "dummy_most_frequent",
            "augmentation": "none",
            "normalization": "none",
            "loss": "not_applicable",
        }
        with self.assertRaisesRegex(ValueError, "stage"):
            CandidateSpec(stage="stage3", adapter="classical", **common)
        with self.assertRaisesRegex(ValueError, "adapter"):
            CandidateSpec(stage="stage2", adapter="unknown", **common)

    def test_experiment_config_rejects_duplicate_or_negative_seeds(self) -> None:
        from cryinsight.experiments.contracts import ExperimentConfig

        with self.assertRaisesRegex(ValueError, "seeds"):
            ExperimentConfig(
                schema_version="1.0",
                wave="A",
                seeds=(42, 42),
                selection_metric="oof_macro_f1",
                candidates=("stage2_majority",),
            )
        with self.assertRaisesRegex(ValueError, "seeds"):
            ExperimentConfig(
                schema_version="1.0",
                wave="A",
                seeds=(-1,),
                selection_metric="oof_macro_f1",
                candidates=("stage2_majority",),
            )


if __name__ == "__main__":
    unittest.main()
