# CryInsight Short-Audio Feature Contract Design

Date: 2026-08-16  
Status: approved design amendment

## Problem and evidence

Stage 1 stopped before its first training epoch because silence trimming reduced a
valid, non-zero ESC-50 transient to seven MFCC frames. The frozen Delta and
Delta-Delta calculation uses a nine-frame window, so feature extraction raised an
`AudioContractError` before the later 128-frame feature padding could run.

A read-only scan of all 3,311 eligible Stage 1 originals found three affected
`not_baby` records, no decode failures, and no zero-energy records:

- `not_baby/3-216284-A-39.wav`: 3,072 trimmed samples, 7 frames
- `not_baby/3-216281-A-39.wav`: 3,072 trimmed samples, 7 frames
- `not_baby/2-83688-A-34.wav`: 3,584 trimmed samples, 8 frames

The legacy trainer caught every exception and silently skipped the entire record.
That behavior is rejected because it changes the training cohort without recording
the exclusion or preserving the target-based augmentation contract.

## Decision

The shared feature extractor will right-zero-pad every valid, finite, non-zero
waveform to the minimum length required by the nine-frame Delta window before MFCC,
Log-Mel, or Chroma extraction. With centered feature framing and a hop length of 512,
the minimum is:

```text
(delta_width - 1) * hop_length = (9 - 1) * 512 = 4,096 samples
```

Right padding is selected because the existing feature contract already right-pads
short feature matrices to 128 frames. It preserves the temporal position of the
observed transient and changes only the previously undefined pre-Delta boundary.
The rule applies identically to originals, augmented derivatives, Stage 1, Stage 2,
and direct shared-extractor use.

Empty, non-finite, or zero-energy waveforms remain errors. The change does not turn
invalid audio into an eligible sample and does not alter labels, hashes, groups,
fold assignments, augmentation targets, model topology, or metric calculation.

## Versioned contract and diagnostics

`PreprocessingConfig` will expose the Delta window and short-audio policy, and its
serialized payload will include the derived minimum waveform sample count. The
preprocessing version will advance from `cryinsight_features_v1` to
`cryinsight_features_v2` so a model bundle cannot claim the old behavior while using
the new one.

Fold materialization errors will add the original record or augmentation sample ID
and source path while preserving the original exception as the cause. This makes a
future audio failure traceable without silently skipping or reducing a class.

## Alternatives rejected

1. Excluding the three records would discard valid environmental transients and
   change the eligible OOF cohort solely because of an implementation boundary.
2. Selecting a different Delta width per record would make feature semantics depend
   on clip length and weaken train/inference comparability.
3. Restoring the legacy catch-and-skip behavior would violate the audited cohort and
   target-count requirements.

## Test and verification design

The change will follow a red-green regression cycle:

1. Add a test proving that a valid 3,072-sample waveform currently fails but must
   produce a finite `float32` tensor of shape `(196, 128, 1)` under the new contract.
2. Add contract assertions for Delta width 9, the 4,096-sample minimum, right-zero-pad
   policy, and preprocessing version 2.
3. Implement only the shared minimum-padding rule and contextual error wrapping.
4. Run the focused regression test, all unit tests, Python compilation, dependency
   checks, both TensorFlow model save/load forward-pass smoke tests, and a read-only
   feature check of all three known short records.

Full five-fold training is not part of the code-fix verification. The failed run
`20260815T181927Z_571bfde3` remains immutable and incomplete; it is not resumed or
deleted. After verification, the same Stage 1 `--train` command creates a new run ID.
