# CryInsight Corrected Two-Stage Training Design

## Source integration and implementation scope

The active publication trainers are
`Models_dbl/Main/train_main_dbl.py` and
`Models_dbl/binary/train_binary_dbl.py`. The files under `models_dbl_OLD` are
read-only references for the existing CNN/BiLSTM/Attention architectures, feature
composition, label orders, and pre-specified training hyperparameters. Legacy model
files, logs, post-hoc reports, and reported 99.x% values are not copied into the
active publication paths and are not reused as corrected results.

Both active entry points may use the shared `cryinsight.audio` and
`cryinsight.training` modules. Shared preprocessing, fold assignment, augmentation,
normalization, OOF validation, and artefact identity are preferred over duplicating
those rules in two standalone scripts. Changes outside these active trainers and
shared modules are out of scope until the corrected Stage 1 and Stage 2 runs finish.

The short-audio boundary discovered during the first Stage 1 attempt is governed by
`2026-08-16-short-audio-feature-contract-design.md`: valid non-zero transients are
right-zero-padded to the minimum waveform length required by the frozen nine-frame
Delta window. The feature contract version advances so the new behavior is recorded
with every fold bundle. A failed training run must write an explicit immutable
`incomplete` verification status with error context; it is never resumed or silently
converted into a successful result.

## Scope and safety

This change corrects the Stage 2 and Stage 1 training protocols without deleting, moving, or overwriting existing model artefacts. Existing top-level models, normalizers, logs, reports, and confusion matrices remain legacy evidence. Every new execution writes to a new immutable `runs/<run_id>/` directory and fails if that directory already exists.

Full five-fold model training starts only after the implementation regression,
dependency, model-serialization, and real-audio checks pass. The user authorized the
Stage 1 and Stage 2 full runs on 2026-08-16. No pre-existing metric is reused as an
estimate of corrected accuracy.

## Data cohort and identity

Both stages discover original `.wav` files from `D:/INFANT CRY/data_set_dbl`, not from the legacy 80/20 copy. Every file receives a SHA-256 hash and a stable record identifier before fold assignment.

Stage 2 uses the five operational corpus labels in this fixed order:

```text
belly_pain, burping, discomfort, hungry, tired
```

Exact-content duplicates with one label retain one deterministic canonical record and mark the remaining copies excluded. Exact-content groups carrying multiple Stage 2 labels are entirely excluded as label conflicts. With no subject/session manifest in the repository, the reliable Stage 2 group is the exact-content SHA-256 group; the resulting evaluation is explicitly limited to grouped clip-level internal validation.

Stage 1 maps eligible infant records to `baby` and ESC-50 records to `not_baby`. ESC-50 target 20 (`crying_baby`) is excluded from the negative class. ESC-50 filenames are parsed into their official fold, source-file, take, and target fields; all fragments from one ESC-50 source file share one group. Infant exact duplicates are canonicalized and grouped by SHA-256. The Stage 1 label order is fixed as `not_baby, baby`.

The audit of every candidate and exclusion is written separately from the eligible fold-assignment manifest so exclusions remain traceable.

## Fold and leakage protocol

The splitter is selected in this order:

1. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` when reliable groups and per-class group support permit it.
2. `GroupKFold(n_splits=5)` when group isolation is feasible but stratification fails.
3. `StratifiedKFold` only when no reliable group relation exists; this fallback is not expected for the audited cohorts and must be recorded as a limitation.

Fold assignments are created from eligible originals before audio augmentation. Each fold asserts zero record, group, and hash overlap between Training and Validation. Each original record is assigned to validation exactly once. Validation contains originals only.

## Target-based augmentation and features

For each fold, the target is the maximum original Training count across labels. Every Training original is retained. Minority labels receive exactly `target - original_count` virtual waveform derivatives, allocated with deterministic shuffled round-robin source selection. Each derivative records its source, transformation, parameters, index, seed, and virtual output identifier. Validation receives zero derivatives.

The frozen feature contract is:

```text
sample rate        22050 Hz
mono               true
silence trim       top_db=20
waveform normalize librosa.util.normalize
n_fft              2048
hop_length         512
channel order      MFCC40, Delta40, DeltaDelta40, LogMel64, Chroma12
frame handling     right zero-pad or truncate to 128
dtype              float32
tensor shape       (196, 128, 1)
```

Training and both stage scripts call the same feature implementation. Stage 2 keeps deterministic post-normalization Mixup as a versioned, configurable regularizer with the legacy default of 500 samples; Stage 1 defaults to no Mixup. Class weighting is disabled after exact target balancing instead of stacking three balancing mechanisms automatically.

## Per-fold training and artefacts

Each fold fits a scalar mean/std normalizer from that fold's exact original-plus-augmented Training feature tensor only. The fold Validation tensor is transformed with that matching normalizer. The normalizer metadata records fit population, axis, shape, dtype, epsilon, feature contract, run ID, and fold ID.

`ModelCheckpoint`, `EarlyStopping`, selected epoch, fold predictions, and fold metrics all use `val_loss`. After fitting, evaluation reloads the selected checkpoint instead of relying on the final in-memory epoch. The CNN/BiLSTM/Attention architectures remain unchanged except for serialization and deterministic-seed support.

Each fold directory contains its model, normalizer, labels, preprocessing config, history, augmentation manifest, class counts, validation predictions, metrics, and a hash-bearing fold manifest. The run root contains protocol/config files, the record audit, fold assignments, pooled OOF predictions, fold metrics, pooled metrics, confusion matrix outputs, and verification evidence.

No best fold is presented as the publication result. Publication evaluation is the pooled original-record OOF result from all five matching fold bundles. Stable compatibility paths are not updated by the training command.

## Accuracy reporting

Corrected accuracy is unknown until all five folds finish. The scripts report only metrics calculated from original OOF rows and label them corrected internal validation. They also report balanced accuracy, macro/weighted F1, per-class metrics, ROC-AUC when defined, confusion matrices, per-fold values, mean and standard deviation, and group-bootstrap confidence intervals. Legacy CV and 20% results are never used as estimates of the corrected result.

## Interfaces and failure behavior

Both scripts expose:

```text
--audit-only      discover, hash, and report the candidate cohort without splitting/training
--prepare-only    create an immutable run, assignments, counts, augmentation plans, and integrity evidence without model training
--run-id          explicit immutable run identifier; generated when omitted
--seed            deterministic seed, default 42
--epochs          default 200
--batch-size      default 32
```

Stage 2 additionally exposes `--mixup-samples`, default 500. Run-directory collision, insufficient groups, missing labels, cross-partition overlap, failed augmentation, non-finite features, invalid tensor shape, incomplete OOF coverage, or artefact hash mismatch stops the run and records it incomplete. No exception is converted into a metric or normal prediction.

## Verification

Pure protocol tests cover ESC-50 parsing/exclusion, duplicate/conflict resolution,
target counts, deterministic augmentation plans, overlap assertions, and exact-once
OOF coverage. The verified Python 3.10 environment contains TensorFlow 2.15.0,
librosa 0.11.0, scikit-learn 1.7.2, NumPy 1.26.4, and matplotlib 3.10.8. Compilation,
39 unit tests, real feature extraction, and both model save/load forward-pass smoke
checks passed before the short-audio failure was found. Those checks must run again
after the feature-contract repair before either authorized full run starts.
