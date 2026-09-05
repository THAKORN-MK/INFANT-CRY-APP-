"""Shared TensorFlow adapter and model/loss factories for neural experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cryinsight.audio.features import (
    PreprocessingConfig,
    extract_fold_tensors,
    mixup_batch,
)
from cryinsight.models.attention import attention_layer_class
from cryinsight.models.stage2_model import build_stage2_model
from cryinsight.training.artefacts import (
    apply_normalizer,
    fit_normalizer,
    save_normalizer,
    sha256_file,
)
from cryinsight.training.checkpoint_staging import CheckpointStaging
from cryinsight.training.feature_cache import FeatureCache
from cryinsight.training.protocol import (
    build_target_augmentation_plan,
    write_json_atomic,
)

from .contracts import CandidateSpec, FoldRequest, FoldResult
from .feature_views import (
    log_mel_view,
    preprocessing_config_for_candidate,
    select_feature_blocks,
)
from .registry import ExperimentProtocolError


@dataclass(frozen=True)
class LossBundle:
    loss: Any
    class_weight: Mapping[int, float] | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class NeuralFoldTensors:
    train_features: np.ndarray
    train_labels: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    train_sample_kinds: tuple[str, ...]
    validation_features: np.ndarray
    validation_labels: tuple[str, ...]
    validation_record_ids: tuple[str, ...]
    validation_sample_kinds: tuple[str, ...]
    augmentation_manifest: tuple[Mapping[str, Any], ...] = ()


TensorBuilder = Callable[[FoldRequest], NeuralFoldTensors]


def effective_number_class_weights(
    counts: Mapping[str, int],
    *,
    beta: float,
) -> dict[str, float]:
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be within [0, 1)")
    if not counts or any(int(value) <= 0 for value in counts.values()):
        raise ValueError("Every class count must be positive")
    raw = {
        str(label): (1.0 - beta) / (1.0 - beta ** int(count))
        if beta > 0.0
        else 1.0
        for label, count in counts.items()
    }
    mean = float(sum(raw.values()) / len(raw))
    result = {label: float(value / mean) for label, value in raw.items()}
    if not all(np.isfinite(value) and value > 0.0 for value in result.values()):
        raise ValueError("Effective-number class weights are invalid")
    return result


def categorical_focal_loss(*, gamma: float, alpha: Sequence[float], tf: Any):
    gamma_value = float(gamma)
    if gamma_value < 0.0:
        raise ValueError("Focal gamma cannot be negative")
    alpha_values = np.asarray(alpha, dtype=np.float32)
    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("Focal alpha must be a non-empty vector")
    alpha_tensor = tf.constant(alpha_values, dtype=tf.float32)

    def loss(y_true, y_pred):
        true_values = tf.cast(y_true, tf.float32)
        probabilities = tf.clip_by_value(
            tf.cast(y_pred, tf.float32),
            tf.constant(1e-7, dtype=tf.float32),
            tf.constant(1.0 - 1e-7, dtype=tf.float32),
        )
        cross_entropy = -true_values * tf.math.log(probabilities)
        modulation = tf.pow(1.0 - probabilities, gamma_value)
        return tf.reduce_sum(alpha_tensor * modulation * cross_entropy, axis=-1)

    loss.__name__ = f"categorical_focal_loss_gamma_{gamma_value:g}".replace(".", "_")
    return loss


def build_loss(
    candidate: CandidateSpec,
    class_counts: Mapping[str, int],
    label_order: Sequence[str],
    tf: Any,
) -> LossBundle:
    labels = tuple(str(label) for label in label_order)
    if set(class_counts) != set(labels):
        raise ExperimentProtocolError("Loss class counts do not match label order")
    label_smoothing = float(candidate.parameters.get("label_smoothing", 0.0))
    if candidate.loss == "categorical_crossentropy":
        return LossBundle(
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing
            ),
            class_weight=None,
            metadata={
                "name": "categorical_crossentropy",
                "label_smoothing": label_smoothing,
            },
        )
    weights = effective_number_class_weights(class_counts, beta=0.999)
    if candidate.loss == "class_balanced_crossentropy":
        return LossBundle(
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing
            ),
            class_weight={index: weights[label] for index, label in enumerate(labels)},
            metadata={
                "name": "class_balanced_crossentropy",
                "beta": 0.999,
                "weights": weights,
                "label_smoothing": label_smoothing,
            },
        )
    if candidate.loss == "focal":
        gamma = float(candidate.parameters.get("focal_gamma", 2.0))
        alpha = [weights[label] for label in labels]
        return LossBundle(
            loss=categorical_focal_loss(gamma=gamma, alpha=alpha, tf=tf),
            class_weight=None,
            metadata={
                "name": "categorical_focal_loss",
                "gamma": gamma,
                "beta": 0.999,
                "alpha": dict(zip(labels, alpha, strict=True)),
            },
        )
    raise ExperimentProtocolError(f"Unsupported neural loss: {candidate.loss}")


def build_small_cnn(tf: Any, input_shape: Sequence[int], num_classes: int):
    layers = tf.keras.layers
    inputs = layers.Input(shape=tuple(input_shape), name="audio_features")
    x = layers.Conv2D(24, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(48, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classifier",
    )(x)
    return tf.keras.Model(inputs, outputs, name="experiment_logmel_small_cnn")


def build_cnn_only(tf: Any, input_shape: Sequence[int], num_classes: int):
    layers = tf.keras.layers
    inputs = layers.Input(shape=tuple(input_shape), name="audio_features")
    x = inputs
    for index, (filters, pool) in enumerate(
        ((32, (2, 2)), (64, (2, 2)), (96, (2, 1))),
        start=1,
    ):
        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            activation="relu",
            name=f"cnn_{index}",
        )(x)
        x = layers.MaxPooling2D(pool, name=f"pool_{index}")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classifier",
    )(x)
    return tf.keras.Model(inputs, outputs, name="experiment_stage2_cnn_only")


def build_cnn_bilstm(tf: Any, input_shape: Sequence[int], num_classes: int):
    layers = tf.keras.layers
    inputs = layers.Input(shape=tuple(input_shape), name="audio_features")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Permute((2, 1, 3), name="time_major")(x)
    x = layers.Reshape((int(x.shape[1]), int(x.shape[2]) * int(x.shape[3])))(x)
    x = layers.Bidirectional(layers.LSTM(64), name="bilstm")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classifier",
    )(x)
    return tf.keras.Model(inputs, outputs, name="experiment_stage2_cnn_bilstm")


def build_neural_model(
    candidate: CandidateSpec,
    input_shape: Sequence[int],
    num_classes: int,
    tf: Any,
):
    if candidate.model == "small_cnn":
        return build_small_cnn(tf, input_shape, num_classes)
    if candidate.model == "cnn_only":
        return build_cnn_only(tf, input_shape, num_classes)
    if candidate.model == "cnn_bilstm":
        return build_cnn_bilstm(tf, input_shape, num_classes)
    if candidate.model == "cnn_bilstm_attention":
        architecture = str(
            candidate.parameters.get("architecture", "corrected_single_branch")
        )
        if architecture == 'corrected_multi_branch' and candidate.feature_view == 'feature_block_subset':
            return _build_subset_multi_branch(candidate, input_shape, num_classes, tf)
        return build_stage2_model(
            tf,
            input_shape,
            num_classes,
            architecture=architecture,
        )
    raise ExperimentProtocolError(f"Unsupported neural model: {candidate.model}")


def _build_subset_multi_branch(candidate: CandidateSpec, input_shape: Sequence[int], num_classes: int, tf: Any):
    """Experiment-local branch removal; the protected primary builder is unchanged."""
    from cryinsight.models.stage2_model import _conv_block, _as_time_sequence, _crop_feature_block

    layers = tf.keras.layers
    inputs = layers.Input(shape=tuple(input_shape), name='audio_features')
    _, boundaries = _feature_metadata(candidate, preprocessing_config_for_candidate(candidate), int(input_shape[0]))
    sequences = []
    families = (('mfcc', {'mfcc', 'delta', 'delta2'}, ((2, 2), (2, 2), (2, 1))),
                ('mel', {'log_mel'}, ((2, 2), (2, 2), (2, 1))),
                ('chroma', {'chroma'}, ((2, 2), (2, 2))))
    for prefix, names, pools in families:
        pieces = [_crop_feature_block(layers, inputs, start, end, f'{prefix}_{name}') for name, start, end in boundaries if name in names]
        if not pieces:
            continue
        x = pieces[0] if len(pieces) == 1 else layers.Concatenate(axis=1, name=f'{prefix}_features')(pieces)
        for index, pool in enumerate(pools, 1):
            x = _conv_block(layers, x, 32 * min(index, 3), pool, index, prefix=prefix, two_convs=index < 3)
        sequences.append(_as_time_sequence(layers, x, f'{prefix}_time_sequence'))
    if not sequences:
        raise ExperimentProtocolError('Multi-branch subset contains no feature branch')
    x = sequences[0] if len(sequences) == 1 else layers.Concatenate(axis=-1, name='time_sequence')(sequences)
    for index, units in enumerate((128, 64), 1):
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True), name=f'bilstm_{index}')(x)
        x = layers.Dropout(.30)(x)
    x = attention_layer_class(tf)(name='temporal_attention')(x)
    for units, dropout in ((256, .40), (128, .40), (64, .30)):
        x = layers.Dense(units, activation='relu')(x)
        if units != 64:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='classifier')(x)
    return tf.keras.Model(inputs, outputs, name='Stage2_corrected_multi_branch_subset_TimeMajor_BiLSTM_Attention')


def _transform_feature_batch(
    values: np.ndarray,
    candidate: CandidateSpec,
) -> np.ndarray:
    features = np.asarray(values, dtype=np.float32)
    if candidate.feature_view in {"all_blocks", "multi_branch_blocks"}:
        return features
    if candidate.feature_view == "log_mel":
        return np.stack([log_mel_view(row) for row in features]).astype(np.float32)
    if candidate.feature_view == "feature_block_subset":
        blocks = tuple(candidate.parameters.get("blocks", ()))
        return np.stack(
            [select_feature_blocks(row, blocks) for row in features]
        ).astype(np.float32)
    raise ExperimentProtocolError(
        f"Feature view {candidate.feature_view!r} is not a neural tensor view"
    )


def _default_tensor_builder(request: FoldRequest) -> NeuralFoldTensors:
    config = request.runtime.get("preprocessing_config")
    if config is None:
        config = preprocessing_config_for_candidate(request.candidate)
    if not isinstance(config, PreprocessingConfig):
        raise ExperimentProtocolError(
            "runtime.preprocessing_config must be a PreprocessingConfig"
        )
    cache = request.runtime.get("feature_cache")
    if cache is not None and not isinstance(cache, FeatureCache):
        raise ExperimentProtocolError("runtime.feature_cache must be a FeatureCache")
    if request.candidate.augmentation == "none":
        augmentation_rows = ()
    elif request.candidate.augmentation in {"waveform_only", "waveform_plus_mixup"}:
        augmentation_rows = build_target_augmentation_plan(
            request.train_records,
            fold=request.fold,
            seed=request.seed,
        ).rows
    else:
        raise ExperimentProtocolError(
            f"Unsupported augmentation: {request.candidate.augmentation}"
        )
    tensors = extract_fold_tensors(
        request.train_records,
        request.validation_records,
        augmentation_rows,
        config=config,
        feature_cache=cache,
    )
    return NeuralFoldTensors(
        train_features=_transform_feature_batch(
            tensors.train_features,
            request.candidate,
        ),
        train_labels=tensors.train_labels,
        train_sample_ids=tensors.train_sample_ids,
        train_sample_kinds=tensors.train_sample_kinds,
        validation_features=_transform_feature_batch(
            tensors.validation_features,
            request.candidate,
        ),
        validation_labels=tensors.validation_labels,
        validation_record_ids=tensors.validation_record_ids,
        validation_sample_kinds=tensors.validation_sample_kinds,
        augmentation_manifest=tuple({
            'sample_id': row.sample_id, 'source_record_id': row.original_record_id,
            'group_id': next(record.group_id for record in request.train_records if record.record_id == row.original_record_id),
            'type': row.augmentation_type, 'parameters': json.loads(row.augmentation_params_json),
            'seed': row.seed, 'fold': request.fold, 'partition': 'train',
        } for row in augmentation_rows),
    )


def _one_hot(labels: Sequence[str], label_order: tuple[str, ...]) -> np.ndarray:
    index = {label: position for position, label in enumerate(label_order)}
    try:
        values = [index[str(label)] for label in labels]
    except KeyError as exc:
        raise ExperimentProtocolError(f"Unknown training label: {exc.args[0]}") from exc
    result = np.zeros((len(values), len(label_order)), dtype=np.float32)
    result[np.arange(len(values)), values] = 1.0
    return result


def _feature_metadata(
    candidate: CandidateSpec,
    config: PreprocessingConfig,
    feature_bins: int,
) -> tuple[tuple[str, ...], tuple[tuple[str, int, int], ...]]:
    if candidate.feature_view in {"all_blocks", "multi_branch_blocks"}:
        return config.feature_order, config.feature_blocks
    if candidate.feature_view == "log_mel":
        return ("log_mel",), (("log_mel", 0, feature_bins),)
    blocks = tuple(str(value) for value in candidate.parameters.get("blocks", ()))
    widths = {"mfcc": 40, "delta": 40, "delta2": 40, "log_mel": 64, "chroma": 12}
    rows: list[tuple[str, int, int]] = []
    start = 0
    for block in blocks:
        end = start + widths[block]
        rows.append((block, start, end))
        start = end
    if start != feature_bins:
        raise ExperimentProtocolError("Feature block metadata does not match tensor bins")
    return blocks, tuple(rows)


class NeuralAdapter:
    def __init__(
        self,
        *,
        tensor_builder: TensorBuilder | None = None,
        tf_module: Any | None = None,
    ):
        self.tensor_builder = tensor_builder or _default_tensor_builder
        self.tf_module = tf_module

    def _tf(self):
        if self.tf_module is not None:
            return self.tf_module
        import tensorflow as tf

        return tf

    def fit_predict_fold(self, request: FoldRequest) -> FoldResult:
        tf = self._tf()
        if request.candidate.adapter != "neural":
            raise ExperimentProtocolError("NeuralAdapter received a non-neural candidate")
        labels = tuple(request.label_order)
        if not labels or len(labels) != len(set(labels)):
            raise ExperimentProtocolError("Fold label order must be unique and non-empty")
        tensors = self.tensor_builder(request)
        if set(tensors.validation_sample_kinds) != {"original"}:
            raise ExperimentProtocolError("Neural validation tensors must be original-only")
        if tuple(tensors.validation_record_ids) != tuple(
            record.record_id for record in request.validation_records
        ):
            raise ExperimentProtocolError("Validation tensor IDs do not match frozen fold")
        if tensors.train_features.ndim != 4 or tensors.validation_features.ndim != 4:
            raise ExperimentProtocolError("Neural features must be rank-4 tensors")
        if tensors.train_features.shape[1:] != tensors.validation_features.shape[1:]:
            raise ExperimentProtocolError("Neural train/validation feature shapes differ")
        original_mask = np.asarray([kind == 'original' for kind in tensors.train_sample_kinds])
        if len(original_mask) != len(tensors.train_features) or len(tensors.train_sample_ids) != len(original_mask) or len(tensors.train_labels) != len(original_mask):
            raise ExperimentProtocolError('Training tensor metadata lengths differ')
        original_ids = tuple(sample_id for sample_id, original in zip(tensors.train_sample_ids, original_mask) if original)
        if original_ids != tuple(record.record_id for record in request.train_records):
            raise ExperimentProtocolError('Training original IDs/order differ from frozen fold')
        originals = {record.record_id: record for record in request.train_records}
        augmented_ids = {sample_id for sample_id, kind in zip(tensors.train_sample_ids, tensors.train_sample_kinds) if kind == 'augmented'}
        if len(set(tensors.train_sample_ids)) != len(tensors.train_sample_ids) or set(tensors.train_sample_kinds) - {'original', 'augmented'}:
            raise ExperimentProtocolError('Training sample identity/kind invalid')
        if {row.get('sample_id') for row in tensors.augmentation_manifest} != augmented_ids or len(tensors.augmentation_manifest) != len(augmented_ids):
            raise ExperimentProtocolError('Augmented training samples require complete provenance')
        for row in tensors.augmentation_manifest:
            source = originals.get(row.get('source_record_id'))
            if source is None or source.group_id != row.get('group_id') or not row.get('type') or not isinstance(row.get('parameters'), Mapping) or not isinstance(row.get('seed'), int):
                raise ExperimentProtocolError('Augmentation source/group/type/parameters/seed invalid')

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        normalizer_path = output_dir / "normalizer.npy"
        history_path = output_dir / "history.csv"
        model_path = output_dir / "model.keras"
        publication_path = output_dir / "checkpoint_publication.json"
        manifest_path = output_dir / "fold_manifest.json"

        config = request.runtime.get("preprocessing_config")
        if not isinstance(config, PreprocessingConfig):
            config = preprocessing_config_for_candidate(request.candidate)
        feature_order, feature_blocks = _feature_metadata(
            request.candidate,
            config,
            int(tensors.train_features.shape[1]),
        )
        normalizer_axis = (
            "global_scalar"
            if request.candidate.normalization == "global_scalar"
            else "per_feature_bin"
        )
        normalizer = fit_normalizer(
            tensors.train_features[original_mask],
            epsilon=config.normalization_epsilon,
            run_id=request.experiment_run_id,
            fold=request.fold,
            axis=normalizer_axis,
            feature_order=feature_order,
            feature_blocks=feature_blocks,
            preprocessing_version=config.version,
        )
        save_normalizer(normalizer_path, normalizer)
        normalizer_manifest_path = output_dir / 'normalizer_manifest.json'
        write_json_atomic(normalizer_manifest_path, {
            'schema_version': '1.0', 'config_sha256': request.runtime.get('config_sha256'),
            'feature_shape': list(tensors.train_features.shape[1:]), 'dtype': str(tensors.train_features.dtype),
            'feature_blocks': [list(row) for row in feature_blocks], 'fit_record_ids': list(original_ids),
            'normalizer_sha256': sha256_file(normalizer_path),
            'normalizer_metadata_sha256': sha256_file(normalizer_path.with_suffix('.npy.metadata.json')),
        })
        x_train = apply_normalizer(tensors.train_features, normalizer)
        x_validation = apply_normalizer(tensors.validation_features, normalizer)
        y_train = _one_hot(tensors.train_labels, labels)

        if request.candidate.augmentation == "waveform_plus_mixup":
            mixup_samples = int(request.candidate.parameters.get("mixup_samples", 500))
            mixed_x, mixed_y = mixup_batch(
                x_train,
                y_train,
                n_samples=mixup_samples,
                alpha=float(request.candidate.parameters.get("mixup_alpha", 0.3)),
                seed=request.seed + request.fold * 100_000,
            )
            x_fit = np.concatenate((x_train, mixed_x), axis=0)
            y_fit = np.concatenate((y_train, mixed_y), axis=0)
        else:
            mixed_x = np.empty((0, *x_train.shape[1:]), dtype=np.float32)
            x_fit = x_train
            y_fit = y_train

        augmentation_manifest_path = output_dir / 'augmentation_manifest.json'
        mixup_rows = []
        if len(mixed_x):
            mixup_seed = request.seed + request.fold * 100_000
            rng = np.random.default_rng(mixup_seed)
            left = rng.integers(0, len(x_train), size=len(mixed_x))
            right = (left + rng.integers(1, len(x_train), size=len(mixed_x))) % len(x_train)
            alpha = float(request.candidate.parameters.get('mixup_alpha', .3))
            weights = rng.beta(alpha, alpha, size=len(mixed_x)).astype(np.float32)
            sources = {record.record_id: {'source_record_id': record.record_id, 'group_id': record.group_id} for record in request.train_records}
            sources.update({row['sample_id']: {'source_record_id': row['source_record_id'], 'group_id': row['group_id']} for row in tensors.augmentation_manifest})
            for index, (lindex, rindex, weight) in enumerate(zip(left, right, weights)):
                sample_ids = [tensors.train_sample_ids[lindex], tensors.train_sample_ids[rindex]]
                mixup_rows.append({'sample_id': f'mixup_{index}', 'type': 'mixup', 'parameters': {'alpha': alpha, 'weight': float(weight)}, 'seed': mixup_seed, 'source_sample_ids': sample_ids, 'sources': [sources[value] for value in sample_ids], 'partition': 'train', 'fold': request.fold})
        write_json_atomic(augmentation_manifest_path, {'schema_version': '1.0', 'fold': request.fold, 'waveform_rows': list(tensors.augmentation_manifest), 'mixup_rows': mixup_rows})

        class_counts = Counter(record.label for record in request.train_records)
        if set(class_counts) != set(labels):
            raise ExperimentProtocolError("Every neural training fold must contain every class")
        loss_bundle = build_loss(request.candidate, class_counts, labels, tf)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(request.seed + request.fold * 1000)
        model = build_neural_model(
            request.candidate,
            x_fit.shape[1:],
            len(labels),
            tf,
        )
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=float(request.candidate.parameters.get("learning_rate", 1e-3)),
                weight_decay=float(request.candidate.parameters.get("weight_decay", 1e-4)),
            ),
            loss=loss_bundle.loss,
            metrics=["accuracy"],
        )
        epochs = int(request.runtime.get("epochs", 60))
        batch_size = int(request.runtime.get("batch_size", 32))
        verbose = int(request.runtime.get("verbose", 1))
        with CheckpointStaging(
            request.experiment_run_id,
            f"{request.candidate.candidate_id}-seed-{request.seed}-fold-{request.fold}",
            model_path.name,
            staging_root=request.runtime.get("checkpoint_staging_root"),
        ) as staging:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    str(staging.local_path),
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=int(request.runtime.get("early_stopping_patience", 10)),
                    restore_best_weights=False,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    mode="min",
                    factor=0.5,
                    patience=int(request.runtime.get("lr_patience", 4)),
                    min_lr=1e-6,
                ),
                tf.keras.callbacks.CSVLogger(str(history_path), append=False),
            ]
            model.fit(
                x_fit,
                y_fit,
                validation_data=(x_validation, _one_hot(tensors.validation_labels, labels)),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=dict(loss_bundle.class_weight)
                if loss_bundle.class_weight is not None
                else None,
                shuffle=True,
                verbose=verbose,
            )
            publication = staging.publish(model_path)
        write_json_atomic(
            publication_path,
            {
                "schema_version": "1.0",
                "candidate_id": request.candidate.candidate_id,
                "seed": request.seed,
                "fold": request.fold,
                "checkpoint_monitor": "val_loss",
                **publication,
            },
        )
        custom_objects = {
            "AttentionLayer": attention_layer_class(tf),
            "CryInsight>AttentionLayer": attention_layer_class(tf),
        }
        selected = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        probabilities = np.asarray(
            selected.predict(x_validation, batch_size=batch_size, verbose=0),
            dtype=np.float64,
        )
        if probabilities.shape != (len(tensors.validation_labels), len(labels)):
            raise ExperimentProtocolError(
                f"Neural validation probability shape is invalid: {probabilities.shape}"
            )
        write_json_atomic(
            manifest_path,
            {
                "schema_version": "1.0",
                "experiment_run_id": request.experiment_run_id,
                "pipeline_run_id": request.pipeline_run_id,
                "candidate_id": request.candidate.candidate_id,
                "seed": request.seed,
                "fold": request.fold,
                "adapter": "neural",
                "model": request.candidate.model,
                "feature_view": request.candidate.feature_view,
                "augmentation": request.candidate.augmentation,
                "normalization": request.candidate.normalization,
                "loss": dict(loss_bundle.metadata),
                "label_order": list(labels),
                "train_record_ids": [record.record_id for record in request.train_records],
                "normalizer_fit_record_ids": [record.record_id for record in request.train_records],
                "normalizer_fit_sample_ids": list(original_ids),
                "validation_record_ids": list(tensors.validation_record_ids),
                "validation_sample_kinds": list(tensors.validation_sample_kinds),
                "training_original_count": int(
                    sum(kind == "original" for kind in tensors.train_sample_kinds)
                ),
                "training_augmented_count": int(
                    sum(kind == "augmented" for kind in tensors.train_sample_kinds)
                ),
                "mixup_sample_count": len(mixed_x),
                "feature_shape": list(x_train.shape[1:]),
                "parameter_count": int(selected.count_params()),
                "config_sha256": request.runtime.get("config_sha256"),
                "assignment_sha256": request.runtime.get("assignment_sha256"),
                "model_sha256": sha256_file(model_path),
                "normalizer_sha256": sha256_file(normalizer_path),
                "normalizer_metadata_sha256": sha256_file(normalizer_path.with_suffix('.npy.metadata.json')),
                "normalizer_manifest_sha256": sha256_file(normalizer_manifest_path),
                "augmentation_manifest_sha256": sha256_file(augmentation_manifest_path),
                "history_sha256": sha256_file(history_path),
                "checkpoint_publication_sha256": sha256_file(publication_path),
            },
        )
        result = FoldResult(
            candidate_id=request.candidate.candidate_id,
            seed=request.seed,
            fold=request.fold,
            validation_record_ids=tuple(tensors.validation_record_ids),
            true_labels=tuple(tensors.validation_labels),
            probabilities=probabilities,
            model_path=model_path,
            manifest_path=manifest_path,
        )
        del model, selected
        tf.keras.backend.clear_session()
        return result
