"""Shared scikit-learn adapter for experiment fold training."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from cryinsight.audio.features import PreprocessingConfig
from cryinsight.training.artefacts import sha256_file
from cryinsight.training.feature_cache import FeatureCache
from cryinsight.training.protocol import OriginalRecord, write_json_atomic

from .contracts import CandidateSpec, FoldRequest, FoldResult
from .feature_views import build_feature_view, preprocessing_config_for_candidate
from .registry import ExperimentProtocolError


FeatureBuilder = Callable[[OriginalRecord, CandidateSpec, FoldRequest], np.ndarray]


def build_classical_estimator(candidate: CandidateSpec, seed: int):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if candidate.model == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent", random_state=seed)
    if candidate.model == "rbf_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=10.0,
                        gamma="scale",
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if candidate.model == "linear_softmax":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if candidate.model == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(256,),
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ExperimentProtocolError(
        f"Unsupported classical model: {candidate.model}"
    )


def _default_feature_builder(
    record: OriginalRecord,
    candidate: CandidateSpec,
    request: FoldRequest,
) -> np.ndarray:
    config = request.runtime.get("preprocessing_config")
    if config is None:
        config = preprocessing_config_for_candidate(candidate)
    if not isinstance(config, PreprocessingConfig):
        raise ExperimentProtocolError(
            "runtime.preprocessing_config must be a PreprocessingConfig"
        )
    cache = request.runtime.get("feature_cache")
    if cache is not None and not isinstance(cache, FeatureCache):
        raise ExperimentProtocolError("runtime.feature_cache must be a FeatureCache")
    raw_blocks = candidate.parameters.get("blocks")
    blocks = tuple(raw_blocks) if raw_blocks is not None else None
    return build_feature_view(
        record,
        candidate.feature_view,
        config,
        cache,
        blocks=blocks,
    )


def _fitted_classifier(estimator: Any) -> Any:
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None and "classifier" in named_steps:
        return named_steps["classifier"]
    return estimator


def _parameter_count(estimator: Any) -> int:
    classifier = _fitted_classifier(estimator)
    if classifier.__class__.__name__ == "DummyClassifier":
        return 0
    if hasattr(classifier, "support_vectors_"):
        return int(
            np.asarray(classifier.support_vectors_).size
            + np.asarray(classifier.dual_coef_).size
            + np.asarray(classifier.intercept_).size
        )
    if hasattr(classifier, "coef_"):
        return int(
            np.asarray(classifier.coef_).size
            + np.asarray(classifier.intercept_).size
        )
    if hasattr(classifier, "coefs_"):
        return int(
            sum(np.asarray(values).size for values in classifier.coefs_)
            + sum(np.asarray(values).size for values in classifier.intercepts_)
        )
    raise ExperimentProtocolError(
        f"Cannot determine parameter count for {classifier.__class__.__name__}"
    )


def _save_estimator_once(path: Path, estimator: Any) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace experiment estimator: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        joblib.dump(estimator, temporary_name)
        with Path(temporary_name).open("r+b") as handle:
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to replace experiment estimator: {path}")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _aligned_probabilities(
    estimator: Any,
    features: np.ndarray,
    label_order: tuple[str, ...],
) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = tuple(str(value) for value in estimator.classes_)
    if set(classes) != set(label_order):
        raise ExperimentProtocolError(
            f"Fitted estimator classes do not match label order: {classes}"
        )
    aligned = np.zeros((raw.shape[0], len(label_order)), dtype=np.float64)
    for source_index, label in enumerate(classes):
        aligned[:, label_order.index(label)] = raw[:, source_index]
    if not np.isfinite(aligned).all() or np.any(aligned < 0.0) or np.any(aligned > 1.0):
        raise ExperimentProtocolError("Estimator produced invalid probabilities")
    return aligned


class ClassicalAdapter:
    def __init__(
        self,
        *,
        feature_builder: FeatureBuilder | None = None,
        estimator_builder: Callable[[CandidateSpec, int], Any] | None = None,
    ):
        self.feature_builder = feature_builder or _default_feature_builder
        self.estimator_builder = estimator_builder or build_classical_estimator

    def fit_predict_fold(self, request: FoldRequest) -> FoldResult:
        import sklearn

        if request.candidate.adapter != "classical":
            raise ExperimentProtocolError("ClassicalAdapter received a non-classical candidate")
        label_order = tuple(request.label_order)
        if not label_order or len(label_order) != len(set(label_order)):
            raise ExperimentProtocolError("Fold label order must be unique and non-empty")
        train_labels = tuple(str(record.label) for record in request.train_records)
        validation_labels = tuple(
            str(record.label) for record in request.validation_records
        )
        if set(train_labels) != set(label_order):
            raise ExperimentProtocolError(
                "Every training fold must contain every declared class"
            )
        if not set(validation_labels).issubset(label_order):
            raise ExperimentProtocolError("Validation labels are outside the label order")
        train_x = np.stack(
            [
                np.asarray(
                    self.feature_builder(record, request.candidate, request),
                    dtype=np.float32,
                )
                for record in request.train_records
            ]
        )
        validation_x = np.stack(
            [
                np.asarray(
                    self.feature_builder(record, request.candidate, request),
                    dtype=np.float32,
                )
                for record in request.validation_records
            ]
        )
        if train_x.shape[1:] != validation_x.shape[1:]:
            raise ExperimentProtocolError(
                "Training and validation feature views have different shapes"
            )
        if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
            raise ExperimentProtocolError("Classical features contain non-finite values")
        estimator = self.estimator_builder(request.candidate, request.seed)
        estimator.fit(train_x, np.asarray(train_labels))
        probabilities = _aligned_probabilities(estimator, validation_x, label_order)

        output_dir = Path(request.output_dir)
        model_path = output_dir / "model.joblib"
        manifest_path = output_dir / "fold_manifest.json"
        _save_estimator_once(model_path, estimator)
        write_json_atomic(
            manifest_path,
            {
                "schema_version": "1.0",
                "experiment_run_id": request.experiment_run_id,
                "pipeline_run_id": request.pipeline_run_id,
                "candidate_id": request.candidate.candidate_id,
                "seed": request.seed,
                "fold": request.fold,
                "adapter": "classical",
                "model": request.candidate.model,
                "feature_view": request.candidate.feature_view,
                "label_order": list(label_order),
                "train_record_ids": [
                    record.record_id for record in request.train_records
                ],
                "validation_record_ids": [
                    record.record_id for record in request.validation_records
                ],
                "normalizer_fit_record_ids": [
                    record.record_id for record in request.train_records
                ],
                "feature_shape": list(train_x.shape[1:]),
                "parameter_count": _parameter_count(estimator),
                "config_sha256": request.runtime.get("config_sha256"),
                "assignment_sha256": request.runtime.get("assignment_sha256"),
                "model_sha256": sha256_file(model_path),
                "numpy_version": np.__version__,
                "scikit_learn_version": sklearn.__version__,
            },
        )
        return FoldResult(
            candidate_id=request.candidate.candidate_id,
            seed=request.seed,
            fold=request.fold,
            validation_record_ids=tuple(
                record.record_id for record in request.validation_records
            ),
            true_labels=validation_labels,
            probabilities=probabilities,
            model_path=model_path,
            manifest_path=manifest_path,
        )
