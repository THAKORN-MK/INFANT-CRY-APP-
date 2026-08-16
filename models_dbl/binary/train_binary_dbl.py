"""Corrected grouped five-fold training for CryInsight Stage 1.

ESC-50 target 20 (``crying_baby``) is excluded from the environmental class,
and eligible ESC-50 clips are grouped by their original ``source_file``.  Full
training starts only with ``--train``; audit modes do not import the ML stack.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryinsight.audio.features import (  # noqa: E402
    PreprocessingConfig,
    extract_fold_tensors,
    extract_original_tensors,
)
from cryinsight.training.artefacts import (  # noqa: E402
    NormalizerStats,
    OofPrediction,
    aggregate_heldout_metrics,
    aggregate_oof_metrics,
    build_fold_manifest,
    create_run_directory,
    heldout_metrics_payload,
    load_normalizer,
    oof_metrics_payload,
    render_confusion_matrix_png,
    save_normalizer,
    select_final_refit_epoch,
    sha256_file,
    write_augmentation_manifest_csv,
    write_confusion_matrix_csv,
    write_fold_metrics_csv,
    write_incomplete_run_verification,
    write_oof_predictions_csv,
)
from cryinsight.training.protocol import (  # noqa: E402
    CohortResolution,
    OriginalRecord,
    SplitResult,
    assign_grouped_folds,
    assert_exact_oof_coverage,
    assert_fold_integrity,
    build_target_augmentation_plan,
    discover_stage1_candidates,
    resolve_stage1_records,
    reserve_heldout_groups,
    write_fold_assignments_csv,
    write_json_atomic,
    write_record_audit_csv,
)


INFANT_LABELS: tuple[str, ...] = (
    "belly_pain",
    "burping",
    "discomfort",
    "hungry",
    "tired",
)
LABEL_ORDER: tuple[str, ...] = ("not_baby", "baby")
N_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_TRAIN_DATA_DIR = PROJECT_ROOT / "data_set_dbl_split" / "train"
DEFAULT_TEST_DATA_DIR = PROJECT_ROOT / "data_set_dbl_split" / "test"
DEFAULT_STAGE_ROOT = Path(__file__).resolve().parent
CHECKPOINT_MONITOR = "val_loss"
_ATTENTION_LAYER_CLASS: Any | None = None


def _tensorflow():
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required only for --train. Activate the CryInsight "
            "TensorFlow 2.15 environment and rerun the command."
        ) from exc
    return tf


def _attention_layer_class(tf: Any):
    global _ATTENTION_LAYER_CLASS
    if _ATTENTION_LAYER_CLASS is not None:
        return _ATTENTION_LAYER_CLASS

    @tf.keras.utils.register_keras_serializable(package="CryInsight")
    class AttentionLayer(tf.keras.layers.Layer):
        def build(self, input_shape):
            width = int(input_shape[-1])
            self.W = self.add_weight(
                shape=(width, width),
                initializer="glorot_uniform",
                trainable=True,
                name="attn_W",
            )
            self.b = self.add_weight(
                shape=(width,),
                initializer="zeros",
                trainable=True,
                name="attn_b",
            )
            self.u = self.add_weight(
                shape=(width,),
                initializer="glorot_uniform",
                trainable=True,
                name="attn_u",
            )
            super().build(input_shape)

        def call(self, inputs):
            score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            score = tf.tensordot(score, self.u, axes=1)
            weights = tf.nn.softmax(score, axis=1)
            return tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)

        def get_config(self):
            return super().get_config()

    _ATTENTION_LAYER_CLASS = AttentionLayer
    return AttentionLayer


def build_binary_model(input_shape: Sequence[int], num_classes: int = 2):
    """Build the audited legacy Stage 1 CNN/BiLSTM/Attention topology."""

    tf = _tensorflow()
    layers = tf.keras.layers
    AttentionLayer = _attention_layer_class(tf)

    inputs = layers.Input(shape=tuple(input_shape))
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    shape = x.shape
    x = layers.Reshape((int(shape[1]), int(shape[2]) * int(shape[3])))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.30)(x)
    x = AttentionLayer()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(
        inputs,
        outputs,
        name="Binary_CNN_MFCC_BiLSTM_Attention",
    )


def audit_stage1(data_dir: Path) -> tuple[CohortResolution, dict[str, Any]]:
    infant_candidates, esc50_candidates = discover_stage1_candidates(
        data_dir,
        infant_labels=INFANT_LABELS,
    )
    resolution = resolve_stage1_records(infant_candidates, esc50_candidates)
    eligible_counts = Counter(row.label for row in resolution.eligible)
    exclusion_counts = Counter(
        row.exclusion_reason for row in resolution.audit if row.status == "excluded"
    )
    target20_count = exclusion_counts.get("esc50_crying_baby", 0)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "stage1_binary_baby_gate",
        "data_root": str(data_dir.resolve()),
        "candidate_original_count": len(infant_candidates) + len(esc50_candidates),
        "candidate_infant_count": len(infant_candidates),
        "candidate_esc50_count": len(esc50_candidates),
        "eligible_original_count": len(resolution.eligible),
        "eligible_counts_by_label": {
            label: eligible_counts[label] for label in LABEL_ORDER
        },
        "excluded_count": (
            len(infant_candidates) + len(esc50_candidates) - len(resolution.eligible)
        ),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "esc50_target20_excluded_count": target20_count,
        "esc50_target20_rule": "exclude_crying_baby_from_not_baby",
        "group_rules": {
            "baby": "exact_content_sha256",
            "not_baby": "esc50_source_file",
        },
        "grouping_limitation": (
            "Infant subject/session identifiers were not verified; infant grouping "
            "prevents exact-content families. ESC-50 negatives use source_file groups."
        ),
        "source_confounding_limitation": (
            "Baby positives originate from InfantCry-DBL while not_baby negatives "
            "originate from ESC-50; recording-source cues may confound Stage 1."
        ),
        "accuracy_status": (
            "Corrected OOF accuracy is not available until all five folds are trained."
        ),
        "evaluation_scope": "model_development_and_corrected_internal_validation",
        "independent_external_validation_performed": False,
    }
    return resolution, payload


def _source_snapshot() -> dict[str, Any]:
    files = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "cryinsight" / "training" / "protocol.py",
        PROJECT_ROOT / "cryinsight" / "training" / "artefacts.py",
        PROJECT_ROOT / "cryinsight" / "audio" / "features.py",
    )
    hashes = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files}
    digest = __import__("hashlib").sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"identifier": f"source_sha256:{digest}", "files": hashes}


def _create_prepared_run(
    *,
    stage_root: Path,
    run_id: str | None,
    resolution: CohortResolution,
    audit_payload: dict[str, Any],
    heldout_resolution: CohortResolution,
    heldout_audit_payload: dict[str, Any],
    heldout_reservation: dict[str, Any],
    split: SplitResult,
    args: argparse.Namespace,
    preprocessing: PreprocessingConfig,
) -> tuple[Path, dict[str, Any]]:
    run_dir = create_run_directory(stage_root, run_id=run_id)
    source_snapshot = _source_snapshot()
    protocol_payload = {
        "schema_version": "2.0",
        "run_id": run_dir.name,
        "stage": "stage1_binary_baby_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "corrected_grouped_internal_validation",
        "heldout_evaluation_scope": "locked_internal_heldout_test_once_after_final_refit",
        "n_folds": N_FOLDS,
        "splitter_name": split.splitter_name,
        "splitter_reason": split.splitter_reason,
        "split_seed": split.split_seed,
        "group_rules": audit_payload["group_rules"],
        "grouping_limitation": audit_payload["grouping_limitation"],
        "fold_assignment_unit": "eligible_original_record",
        "augmentation_scope": "training_partition_only",
        "normalizer_fit_scope": "fold_training_features_only",
        "checkpoint_monitor": CHECKPOINT_MONITOR,
        "oof_support": "original_validation_records_only_exactly_once",
        "class_weight_enabled": False,
        "mixup_enabled": False,
        "independent_external_validation_performed": False,
        "heldout_test_used_for_model_selection": False,
        "final_refit_epoch_rule": "median_fold_best_epoch",
        "heldout_overlap_policy": "test_priority_remove_overlapping_train_records",
        "heldout_provenance_limitation": (
            "Train and held-out Test come from the same underlying corpus, which "
            "was available during legacy model development; this is not external validation."
        ),
        "source_snapshot_identifier": source_snapshot["identifier"],
    }
    run_config = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "train_data_dir": str(args.train_data_dir.resolve()),
        "test_data_dir": str(args.test_data_dir.resolve()),
        "stage_root": str(args.stage_root.resolve()),
        "labels": list(LABEL_ORDER),
        "seed": args.seed,
        "n_folds": N_FOLDS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "mixup_samples": 0,
        "class_weight_enabled": False,
        "early_stopping_patience": args.early_stopping_patience,
        "lr_patience": args.lr_patience,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "source_snapshot": source_snapshot,
    }
    write_json_atomic(run_dir / "protocol.json", protocol_payload)
    write_json_atomic(run_dir / "run_config.json", run_config)
    write_json_atomic(run_dir / "dataset_audit.json", audit_payload)
    write_json_atomic(run_dir / "heldout_dataset_audit.json", heldout_audit_payload)
    write_json_atomic(run_dir / "heldout_reservation.json", heldout_reservation)
    write_json_atomic(run_dir / "labels_binary_dbl.json", list(LABEL_ORDER))
    write_json_atomic(run_dir / "preprocessing_config.json", preprocessing.to_dict())
    write_record_audit_csv(run_dir / "record_audit.csv", resolution.audit)
    write_record_audit_csv(
        run_dir / "heldout_record_audit.csv", heldout_resolution.audit
    )
    write_fold_assignments_csv(run_dir / "fold_assignments.csv", split.assignments)
    return run_dir, source_snapshot


def _set_determinism(tf: Any, seed: int) -> dict[str, Any]:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    deterministic_ops = False
    try:
        tf.config.experimental.enable_op_determinism()
        deterministic_ops = True
    except Exception:
        deterministic_ops = False
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "tensorflow_seed": seed,
        "tensorflow_deterministic_ops_enabled": deterministic_ops,
    }


def _one_hot(labels: Sequence[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    try:
        indices = np.asarray([label_to_index[label] for label in labels], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unknown Stage 1 label: {exc.args[0]!r}") from exc
    return np.eye(len(LABEL_ORDER), dtype=np.float32)[indices]


def _fold_records(
    split: SplitResult,
    fold: int,
) -> tuple[tuple[OriginalRecord, ...], tuple[OriginalRecord, ...]]:
    training = tuple(
        assignment.record
        for assignment in split.assignments
        if assignment.validation_fold != fold
    )
    validation = tuple(
        assignment.record
        for assignment in split.assignments
        if assignment.validation_fold == fold
    )
    assert_fold_integrity(training, validation)
    return training, validation


def train_final_model(
    *,
    run_dir: Path,
    source_snapshot: dict[str, Any],
    resolution: CohortResolution,
    heldout_resolution: CohortResolution,
    preprocessing: PreprocessingConfig,
    fold_metric_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Refit Stage 1 on all development data, then open the locked test once."""

    tf = _tensorflow()
    epoch_selection = select_final_refit_epoch(
        fold_metric_rows,
        expected_folds=N_FOLDS,
    )
    final_epoch = int(epoch_selection["final_epoch"])
    deployment_dir = run_dir / "best_model"
    deployment_dir.mkdir(exist_ok=False)
    model_path = deployment_dir / "best_model_binary_dbl.keras"
    normalizer_path = deployment_dir / "norm_stats_binary_dbl.npy"
    labels_path = deployment_dir / "labels_binary_dbl.json"
    preprocessing_path = deployment_dir / "preprocessing_config.json"
    augmentation_path = deployment_dir / "augmentation_manifest.csv"
    class_counts_path = deployment_dir / "class_counts.json"
    history_path = deployment_dir / "history.csv"
    final_refit_manifest_path = deployment_dir / "final_refit_manifest.json"
    deployment_manifest_path = deployment_dir / "deployment_manifest.json"

    augmentation_plan = build_target_augmentation_plan(
        resolution.eligible,
        fold="final_refit",
        seed=args.seed,
    )
    write_augmentation_manifest_csv(augmentation_path, augmentation_plan.rows)
    write_json_atomic(
        class_counts_path,
        {
            "target_samples_per_class": augmentation_plan.target_samples_per_class,
            "original_training_by_label": augmentation_plan.original_by_label,
            "generated_training_by_label": augmentation_plan.generated_by_label,
            "final_pre_mixup_training_by_label": augmentation_plan.final_by_label,
            "mixup_samples": 0,
            "class_weight_enabled": False,
            "validation_original_count": 0,
        },
    )
    write_json_atomic(labels_path, list(LABEL_ORDER))
    write_json_atomic(preprocessing_path, preprocessing.to_dict())

    tensors = extract_fold_tensors(
        resolution.eligible,
        (),
        augmentation_plan.rows,
        config=preprocessing,
        allow_empty_validation=True,
    )
    mean = float(tensors.train_features.mean(dtype=np.float64))
    std = float(tensors.train_features.std(dtype=np.float64))
    save_normalizer(
        normalizer_path,
        NormalizerStats(
            mean=mean,
            std=std,
            epsilon=preprocessing.normalization_epsilon,
            axis="global_scalar",
            feature_shape=preprocessing.feature_shape,
            dtype="float32",
            fit_sample_count=int(tensors.train_features.shape[0]),
            run_id=run_dir.name,
            fold="final_refit",
        ),
    )
    persisted_normalizer = load_normalizer(
        normalizer_path,
        expected_run_id=run_dir.name,
        expected_fold="final_refit",
    )
    x_train = np.asarray(
        (tensors.train_features - persisted_normalizer.mean)
        / persisted_normalizer.std,
        dtype=np.float32,
    )
    if not np.isfinite(x_train).all():
        raise ValueError("Final-refit normalization produced non-finite values")
    y_train = _one_hot(tensors.train_labels)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed + 100_000)
    model = build_binary_model(
        preprocessing.feature_shape,
        num_classes=len(LABEL_ORDER),
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=args.label_smoothing
        ),
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=final_epoch,
        callbacks=[tf.keras.callbacks.CSVLogger(str(history_path), append=False)],
        shuffle=True,
        verbose=1,
    )
    model.save(str(model_path))

    refit_files = {
        "model": model_path,
        "normalizer": normalizer_path,
        "normalizer_metadata": normalizer_path.with_suffix(
            normalizer_path.suffix + ".metadata.json"
        ),
        "labels": labels_path,
        "preprocessing_config": preprocessing_path,
        "augmentation_manifest": augmentation_path,
        "class_counts": class_counts_path,
        "history": history_path,
    }
    write_json_atomic(
        final_refit_manifest_path,
        {
            "schema_version": "1.0",
            "run_id": run_dir.name,
            "stage": "stage1_binary_baby_gate",
            "artefact_role": "final_refit",
            "training_original_count": len(resolution.eligible),
            "training_augmented_count": len(augmentation_plan.rows),
            "normalizer_fit_scope": "all_final_refit_training_features_only",
            "feature_shape": list(preprocessing.feature_shape),
            "label_order": list(LABEL_ORDER),
            "source_snapshot_identifier": source_snapshot["identifier"],
            **epoch_selection,
            "files": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in refit_files.items()
            },
        },
    )
    write_json_atomic(
        deployment_manifest_path,
        {
            "schema_version": "1.0",
            "status": "ready_for_deployment",
            "stage": "stage1_binary_baby_gate",
            "run_id": run_dir.name,
            "deployment_strategy": "final_refit",
            "deployment_model": str(model_path),
            "deployment_normalizer": str(normalizer_path),
            **epoch_selection,
            "final_refit_manifest": {
                "path": str(final_refit_manifest_path),
                "sha256": sha256_file(final_refit_manifest_path),
            },
            "files": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in {
                    **refit_files,
                    "final_refit_manifest": final_refit_manifest_path,
                }.items()
            },
        },
    )

    AttentionLayer = _attention_layer_class(tf)
    selected_model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "AttentionLayer": AttentionLayer,
            "CryInsight>AttentionLayer": AttentionLayer,
        },
    )
    heldout_tensors = extract_original_tensors(
        heldout_resolution.eligible,
        config=preprocessing,
        partition_name="Locked internal held-out test",
    )
    x_heldout = np.asarray(
        (heldout_tensors.features - persisted_normalizer.mean)
        / persisted_normalizer.std,
        dtype=np.float32,
    )
    if not np.isfinite(x_heldout).all():
        raise ValueError("Final held-out normalization produced non-finite values")
    probabilities = np.asarray(
        selected_model.predict(x_heldout, batch_size=args.batch_size, verbose=0),
        dtype=np.float64,
    )
    if probabilities.shape != (
        len(heldout_resolution.eligible),
        len(LABEL_ORDER),
    ):
        raise RuntimeError(f"Final test prediction shape {probabilities.shape} is invalid")
    predictions = [
        OofPrediction(
            record_id=record.record_id,
            filepath=str(record.filepath),
            label=record.label,
            group_id=record.group_id,
            fold="final_refit",
            predicted_label=LABEL_ORDER[int(np.argmax(scores))],
            scores=tuple(float(score) for score in scores),
            sample_kind="original",
            model_path=str(model_path),
            normalizer_path=str(normalizer_path),
            run_id=run_dir.name,
        )
        for record, scores in zip(
            heldout_resolution.eligible,
            probabilities,
            strict=True,
        )
    ]
    result = aggregate_heldout_metrics(
        heldout_resolution.eligible,
        predictions,
        label_order=LABEL_ORDER,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    predictions_path = run_dir / "final_test_predictions.csv"
    metrics_path = run_dir / "final_test_metrics.json"
    confusion_csv_path = run_dir / "final_test_confusion_matrix.csv"
    confusion_png_path = run_dir / "final_test_confusion_matrix.png"
    manifest_path = run_dir / "final_test_manifest.json"
    write_oof_predictions_csv(predictions_path, predictions, label_order=LABEL_ORDER)
    metrics_payload = heldout_metrics_payload(result)
    metrics_payload.update(
        {
            "run_id": run_dir.name,
            "stage": "stage1_binary_baby_gate",
            "evaluation_count": 1,
            "evaluated_model_role": "final_refit",
            "heldout_test_used_for_model_selection": False,
        }
    )
    write_json_atomic(metrics_path, metrics_payload)
    write_confusion_matrix_csv(
        confusion_csv_path,
        result.confusion_matrix,
        label_order=LABEL_ORDER,
    )
    render_confusion_matrix_png(
        confusion_png_path,
        result.confusion_matrix,
        label_order=LABEL_ORDER,
        title="Stage 1 final locked internal held-out test",
    )
    test_files = {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "confusion_matrix_csv": confusion_csv_path,
        "confusion_matrix_png": confusion_png_path,
    }
    write_json_atomic(
        manifest_path,
        {
            "schema_version": "1.0",
            "run_id": run_dir.name,
            "stage": "stage1_binary_baby_gate",
            "evaluation_scope": "locked_internal_heldout_test_once_after_final_refit",
            "evaluation_count": 1,
            "support_original_records": len(heldout_resolution.eligible),
            "augmentation_applied": False,
            "heldout_test_used_for_model_selection": False,
            "independent_external_validation_performed": False,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "files": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in test_files.items()
            },
        },
    )
    del model, selected_model, tensors, x_train, y_train, x_heldout, probabilities
    tf.keras.backend.clear_session()
    return {
        "deployment_dir": deployment_dir,
        "epoch_selection": epoch_selection,
        "final_test_metrics": metrics_payload,
        "final_test_paths": {**test_files, "manifest": manifest_path},
    }


def run_training(
    *,
    run_dir: Path,
    source_snapshot: dict[str, Any],
    resolution: CohortResolution,
    heldout_resolution: CohortResolution,
    split: SplitResult,
    preprocessing: PreprocessingConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tf = _tensorflow()
    seed_evidence = _set_determinism(tf, args.seed)
    all_oof: list[OofPrediction] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n[Stage 1] Fold {fold}/{N_FOLDS}")
        training, validation = _fold_records(split, fold)
        augmentation_plan = build_target_augmentation_plan(
            training,
            fold=fold,
            seed=args.seed,
        )
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=False)
        augmentation_path = fold_dir / "augmentation_manifest.csv"
        class_counts_path = fold_dir / "class_counts.json"
        labels_path = fold_dir / "labels_binary_dbl.json"
        preprocessing_path = fold_dir / "preprocessing_config.json"
        model_path = fold_dir / f"fold_{fold}_binary_dbl.keras"
        normalizer_path = fold_dir / "norm_stats_binary_dbl.npy"
        history_path = fold_dir / "history.csv"
        predictions_path = fold_dir / "validation_predictions.csv"
        metrics_path = fold_dir / "metrics.json"

        write_augmentation_manifest_csv(augmentation_path, augmentation_plan.rows)
        write_json_atomic(
            class_counts_path,
            {
                "target_samples_per_class": augmentation_plan.target_samples_per_class,
                "original_training_by_label": augmentation_plan.original_by_label,
                "generated_training_by_label": augmentation_plan.generated_by_label,
                "final_pre_mixup_training_by_label": augmentation_plan.final_by_label,
                "mixup_samples": 0,
                "class_weight_enabled": False,
                "validation_original_by_label": dict(
                    sorted(Counter(row.label for row in validation).items())
                ),
            },
        )
        write_json_atomic(labels_path, list(LABEL_ORDER))
        write_json_atomic(preprocessing_path, preprocessing.to_dict())

        tensors = extract_fold_tensors(
            training,
            validation,
            augmentation_plan.rows,
            config=preprocessing,
        )
        mean = float(tensors.train_features.mean(dtype=np.float64))
        std = float(tensors.train_features.std(dtype=np.float64))
        normalizer = NormalizerStats(
            mean=mean,
            std=std,
            epsilon=preprocessing.normalization_epsilon,
            axis="global_scalar",
            feature_shape=preprocessing.feature_shape,
            dtype="float32",
            fit_sample_count=int(tensors.train_features.shape[0]),
            run_id=run_dir.name,
            fold=fold,
        )
        save_normalizer(normalizer_path, normalizer)
        persisted_normalizer = load_normalizer(
            normalizer_path,
            expected_run_id=run_dir.name,
            expected_fold=fold,
        )
        mean = persisted_normalizer.mean
        std = persisted_normalizer.std
        x_train = np.asarray((tensors.train_features - mean) / std, dtype=np.float32)
        x_validation = np.asarray(
            (tensors.validation_features - mean) / std, dtype=np.float32
        )
        if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
            raise ValueError(f"Fold {fold} normalization produced non-finite values")
        y_train = _one_hot(tensors.train_labels)
        y_validation = _one_hot(tensors.validation_labels)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + fold)
        model = build_binary_model(
            preprocessing.feature_shape,
            num_classes=len(LABEL_ORDER),
        )
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=args.label_smoothing
            ),
            metrics=["accuracy"],
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.early_stopping_patience,
                restore_best_weights=False,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.3,
                patience=args.lr_patience,
                min_lr=1e-7,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(history_path), append=False),
        ]
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )
        val_losses = history.history.get("val_loss", [])
        if not val_losses or not model_path.is_file():
            raise RuntimeError(f"Fold {fold} did not produce a selected val_loss checkpoint")
        best_epoch = int(np.argmin(np.asarray(val_losses, dtype=float))) + 1
        best_val_loss = float(val_losses[best_epoch - 1])
        AttentionLayer = _attention_layer_class(tf)
        selected_model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                "AttentionLayer": AttentionLayer,
                "CryInsight>AttentionLayer": AttentionLayer,
            },
        )
        probabilities = np.asarray(
            selected_model.predict(x_validation, batch_size=args.batch_size, verbose=0),
            dtype=np.float64,
        )
        if probabilities.shape != (len(validation), len(LABEL_ORDER)):
            raise RuntimeError(
                f"Fold {fold} prediction shape {probabilities.shape} is invalid"
            )
        fold_oof: list[OofPrediction] = []
        for record, scores in zip(validation, probabilities, strict=True):
            predicted_label = LABEL_ORDER[int(np.argmax(scores))]
            fold_oof.append(
                OofPrediction(
                    record_id=record.record_id,
                    filepath=str(record.filepath),
                    label=record.label,
                    group_id=record.group_id,
                    fold=fold,
                    predicted_label=predicted_label,
                    scores=tuple(float(score) for score in scores),
                    sample_kind="original",
                    model_path=str(model_path),
                    normalizer_path=str(normalizer_path),
                    run_id=run_dir.name,
                )
            )
        fold_result = aggregate_oof_metrics(
            validation,
            fold_oof,
            label_order=LABEL_ORDER,
            bootstrap_iterations=0,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_oof_predictions_csv(
            predictions_path,
            fold_oof,
            label_order=LABEL_ORDER,
        )
        fold_metrics_payload = oof_metrics_payload(fold_result)
        fold_metrics_payload.update(
            {
                "run_id": run_dir.name,
                "fold": fold,
                "checkpoint_monitor": CHECKPOINT_MONITOR,
                "best_epoch": best_epoch,
                "selected_checkpoint_val_loss": best_val_loss,
            }
        )
        write_json_atomic(metrics_path, fold_metrics_payload)
        fold_metric_rows.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "selected_checkpoint_val_loss": best_val_loss,
                **fold_result.pooled_metrics,
            }
        )
        all_oof.extend(fold_oof)

        manifest = build_fold_manifest(
            run_id=run_dir.name,
            fold=fold,
            artefact_paths={
                "model": model_path,
                "normalizer": normalizer_path,
                "normalizer_metadata": normalizer_path.with_suffix(
                    normalizer_path.suffix + ".metadata.json"
                ),
                "labels": labels_path,
                "preprocessing_config": preprocessing_path,
                "fold_assignments": run_dir / "fold_assignments.csv",
                "augmentation_manifest": augmentation_path,
                "class_counts": class_counts_path,
                "history": history_path,
                "validation_predictions": predictions_path,
                "metrics": metrics_path,
            },
            splitter_name=split.splitter_name,
            split_seed=split.split_seed,
            checkpoint_monitor="val_loss",
            best_epoch=best_epoch,
            feature_shape=preprocessing.feature_shape,
            label_order=LABEL_ORDER,
            training_original_count=len(training),
            training_augmented_count=len(augmentation_plan.rows),
            validation_original_count=len(validation),
            source_snapshot_identifier=source_snapshot["identifier"],
        )
        write_json_atomic(fold_dir / "fold_manifest.json", manifest)
        print(
            f"Fold {fold}: selected epoch {best_epoch}, "
            f"val_loss={best_val_loss:.6f}, original validation n={len(validation)}"
        )
        del (
            model,
            selected_model,
            tensors,
            x_train,
            x_validation,
            y_train,
            y_validation,
            probabilities,
        )
        tf.keras.backend.clear_session()

    coverage = assert_exact_oof_coverage(
        [record.record_id for record in resolution.eligible],
        [row.record_id for row in all_oof],
    )
    result = aggregate_oof_metrics(
        resolution.eligible,
        all_oof,
        label_order=LABEL_ORDER,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    oof_predictions_path = run_dir / "oof_predictions.csv"
    oof_metrics_path = run_dir / "oof_metrics.json"
    confusion_csv_path = run_dir / "oof_confusion_matrix.csv"
    confusion_png_path = run_dir / "oof_confusion_matrix.png"
    fold_metrics_path = run_dir / "fold_metrics.csv"
    write_oof_predictions_csv(
        oof_predictions_path,
        all_oof,
        label_order=LABEL_ORDER,
    )
    metrics_payload = oof_metrics_payload(result)
    metrics_payload.update(
        {
            "run_id": run_dir.name,
            "stage": "stage1_binary_baby_gate",
            "oof_coverage": {
                "expected_count": coverage.expected_count,
                "predicted_count": coverage.predicted_count,
                "duplicate_count": coverage.duplicate_count,
                "missing_count": coverage.missing_count,
            },
        }
    )
    write_json_atomic(oof_metrics_path, metrics_payload)
    write_confusion_matrix_csv(
        confusion_csv_path,
        result.confusion_matrix,
        label_order=LABEL_ORDER,
    )
    render_confusion_matrix_png(
        confusion_png_path,
        result.confusion_matrix,
        label_order=LABEL_ORDER,
        title="Stage 1 corrected grouped OOF confusion matrix",
    )
    write_fold_metrics_csv(fold_metrics_path, fold_metric_rows)
    final_result = train_final_model(
        run_dir=run_dir,
        source_snapshot=source_snapshot,
        resolution=resolution,
        heldout_resolution=heldout_resolution,
        preprocessing=preprocessing,
        fold_metric_rows=fold_metric_rows,
        args=args,
    )
    deployment_dir = Path(final_result["deployment_dir"])
    final_test_paths = final_result["final_test_paths"]
    print(f"Stage 1 final-refit Webapp bundle: {deployment_dir}")

    verification_targets = {
        "protocol": run_dir / "protocol.json",
        "run_config": run_dir / "run_config.json",
        "dataset_audit": run_dir / "dataset_audit.json",
        "heldout_dataset_audit": run_dir / "heldout_dataset_audit.json",
        "heldout_reservation": run_dir / "heldout_reservation.json",
        "root_labels": run_dir / "labels_binary_dbl.json",
        "root_preprocessing_config": run_dir / "preprocessing_config.json",
        "record_audit": run_dir / "record_audit.csv",
        "heldout_record_audit": run_dir / "heldout_record_audit.csv",
        "fold_assignments": run_dir / "fold_assignments.csv",
        "oof_predictions": oof_predictions_path,
        "oof_metrics": oof_metrics_path,
        "oof_confusion_matrix_csv": confusion_csv_path,
        "oof_confusion_matrix_png": confusion_png_path,
        "fold_metrics": fold_metrics_path,
        "final_test_predictions": final_test_paths["predictions"],
        "final_test_metrics": final_test_paths["metrics"],
        "final_test_confusion_matrix_csv": final_test_paths[
            "confusion_matrix_csv"
        ],
        "final_test_confusion_matrix_png": final_test_paths[
            "confusion_matrix_png"
        ],
        "final_test_manifest": final_test_paths["manifest"],
        "deployment_model": deployment_dir / "best_model_binary_dbl.keras",
        "deployment_normalizer": deployment_dir / "norm_stats_binary_dbl.npy",
        "deployment_labels": deployment_dir / "labels_binary_dbl.json",
        "deployment_preprocessing": deployment_dir / "preprocessing_config.json",
        "final_refit_manifest": deployment_dir / "final_refit_manifest.json",
        "deployment_manifest": deployment_dir / "deployment_manifest.json",
    }
    for fold in range(1, N_FOLDS + 1):
        verification_targets[f"fold_{fold}_manifest"] = (
            run_dir / f"fold_{fold}" / "fold_manifest.json"
        )
    verification = {
        "schema_version": "1.0",
        "status": "complete",
        "run_id": run_dir.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "folds_completed": N_FOLDS,
        "oof_coverage": metrics_payload["oof_coverage"],
        "seed_evidence": seed_evidence,
        "source_snapshot": source_snapshot,
        "deployment_selection": {
            "strategy": "final_refit",
            **final_result["epoch_selection"],
        },
        "artefact_sha256": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in verification_targets.items()
        },
        "evaluation_scope": "corrected_grouped_internal_validation",
        "heldout_evaluation_scope": "locked_internal_heldout_test_once_after_final_refit",
        "heldout_test_evaluation_count": 1,
        "heldout_test_used_for_model_selection": False,
        "independent_external_validation_performed": False,
    }
    write_json_atomic(run_dir / "verification.json", verification)
    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CryInsight Stage 1 corrected grouped five-fold internal validation"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Hash and audit originals; do not create artefacts or import ML dependencies.",
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Audit, assign five folds, and save immutable manifests; do not train.",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="Run all five corrected folds and produce pooled original OOF metrics.",
    )
    parser.add_argument(
        "--train-data-dir",
        "--data-dir",
        dest="train_data_dir",
        type=Path,
        default=DEFAULT_TRAIN_DATA_DIR,
        help="Training partition (legacy alias: --data-dir).",
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=DEFAULT_TEST_DATA_DIR,
        help="Locked internal held-out test evaluated once after final refit.",
    )
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.bootstrap_iterations < 0:
        parser.error("--bootstrap-iterations cannot be negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.train_data_dir = args.train_data_dir.resolve()
    args.test_data_dir = args.test_data_dir.resolve()
    args.stage_root = args.stage_root.resolve()

    discovered_training, audit_payload = audit_stage1(args.train_data_dir)
    heldout_resolution, heldout_audit_payload = audit_stage1(args.test_data_dir)
    resolution, reservation_report = reserve_heldout_groups(
        discovered_training,
        heldout_resolution.eligible,
    )
    reservation_payload = asdict(reservation_report)
    eligible_counts = Counter(row.label for row in resolution.eligible)
    exclusion_counts = Counter(
        row.exclusion_reason for row in resolution.audit if row.status == "excluded"
    )
    audit_payload.update(
        {
            "eligible_original_count_before_heldout_reservation": len(
                discovered_training.eligible
            ),
            "eligible_original_count": len(resolution.eligible),
            "eligible_counts_by_label": {
                label: eligible_counts[label] for label in LABEL_ORDER
            },
            "excluded_count": audit_payload["candidate_original_count"]
            - len(resolution.eligible),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "heldout_reservation": reservation_payload,
        }
    )
    print(
        json.dumps(
            {
                "training": audit_payload,
                "locked_heldout_test": heldout_audit_payload,
                "heldout_reservation": reservation_payload,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if args.audit_only:
        return 0

    # Assign eligible originals/groups before planning any Training derivative.
    split = assign_grouped_folds(
        resolution.eligible,
        n_folds=N_FOLDS,
        seed=args.seed,
        reliable_groups=True,
    )
    preprocessing = PreprocessingConfig.stage1_binary()
    run_dir, source_snapshot = _create_prepared_run(
        stage_root=args.stage_root,
        run_id=args.run_id,
        resolution=resolution,
        audit_payload=audit_payload,
        heldout_resolution=heldout_resolution,
        heldout_audit_payload=heldout_audit_payload,
        heldout_reservation=reservation_payload,
        split=split,
        args=args,
        preprocessing=preprocessing,
    )
    if args.prepare_only:
        write_json_atomic(
            run_dir / "verification.json",
            {
                "schema_version": "1.0",
                "status": "prepared_only",
                "run_id": run_dir.name,
                "training_started": False,
                "folds_completed": 0,
                "eligible_original_count": len(resolution.eligible),
                "heldout_original_count": len(heldout_resolution.eligible),
                "heldout_reservation": reservation_payload,
                "splitter_name": split.splitter_name,
                "split_seed": split.split_seed,
                "source_snapshot": source_snapshot,
                "accuracy_status": "NOT_RUN",
            },
        )
        print(f"Prepared immutable Stage 1 run: {run_dir}")
        return 0

    try:
        metrics = run_training(
            run_dir=run_dir,
            source_snapshot=source_snapshot,
            resolution=resolution,
            heldout_resolution=heldout_resolution,
            split=split,
            preprocessing=preprocessing,
            args=args,
        )
    except Exception as exc:
        write_incomplete_run_verification(
            run_dir,
            stage="stage1_binary_baby_gate",
            error=exc,
        )
        raise
    print(json.dumps(metrics["pooled_metrics"], ensure_ascii=False, indent=2))
    print(f"Completed corrected Stage 1 run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
