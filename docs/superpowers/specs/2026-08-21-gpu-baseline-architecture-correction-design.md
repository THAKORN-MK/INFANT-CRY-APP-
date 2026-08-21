# GPU, Baseline, and Architecture Correction Design

วันที่: 2026-08-21

สถานะ: รอผู้ใช้ตรวจ written spec ก่อน implementation

ขอบเขตโครงการ: `D:/INFANT CRY`

## 1. วัตถุประสงค์

ปรับ Cry audio classification pipeline ให้:

1. ใช้ TensorFlow GPU ผ่าน WSL2 และ RTX 5070 Ti อย่างตรวจสอบได้
2. รองรับ TensorFlow 2.21, Keras 3 และ Python 3.10 โดยไม่ทำลาย corrected grouped 5-fold protocol
3. แก้การจัดแกน Time/Feature ก่อนเข้า BiLSTM ของ Stage 1 และ Stage 2
4. ลดการสูญเสียข้อมูลเวลาอันเกิดจาก pooling
5. ประเมิน normalization, feature composition และ augmentation ด้วย ablation
6. เพิ่ม Baseline ที่ใช้ fold assignments และ evaluation protocol เดียวกับ proposed model
7. เพิ่มหลาย random seeds, end-to-end cascade metrics และ publication metrics ที่ยังขาด
8. รักษา immutable runs เดิมและรูปแบบ final `best_model` หนึ่งชุดต่อ Stage
9. อัปเดต `README.md`, `Architecture.md` และรายงานให้ตรงกับ implementation และ GPU environment ใหม่

## 2. สิ่งที่อนุมัติและสิ่งที่ยังไม่อยู่ในขอบเขต

### 2.1 อยู่ในขอบเขต

- GPU runtime, device verification และ environment manifest
- TensorFlow 2.21/Keras 3 compatibility
- Stage 1/Stage 2 time-axis correction
- time-preserving pooling
- per-feature-bin หรือ per-feature-block normalization
- multi-branch Stage 2 candidate
- feature cache
- Baseline และ Ablation framework
- repeated seeds
- end-to-end cascade evaluation
- ROC/PR curves และ metrics จาก prediction artefacts
- Tests, README, Architecture และ Report updates

### 2.2 ยังไม่อยู่ในขอบเขตตามคำสั่งผู้ใช้

หัวข้อ “การรองรับเสียงภายนอก” ยังอยู่ระหว่างการตัดสินใจ จึงยังไม่ implement:

- external-audio input contract ใหม่
- automatic multi-window inference สำหรับไฟล์ภายนอก
- quality rejection สำหรับ silence/clipping/external codecs
- webapp routing สำหรับ external audio
- การเปลี่ยนข้อกำหนด input format ของ deployment bundle

การแก้ architecture จะไม่แอบเพิ่มพฤติกรรมเหล่านี้ หากภายหลังอนุมัติจะออกแบบเป็น scope แยก

## 3. สิ่งที่ต้องรักษา

- split original records/groups ก่อน augmentation
- augmentation เฉพาะ training partition
- validation และ OOF support เป็น original records เท่านั้น
- per-fold model, normalizer, labels และ preprocessing metadata
- original development record มี OOF prediction exactly once
- final model เป็น final-refit model ไม่ใช่ best fold
- `best_model_binary_dbl.keras` และ `best_model_main_dbl.keras` มีอย่างละหนึ่งตัวใน `best_model`
- Fold 1-5 artefacts อยู่ใน `fold_1` ถึง `fold_5`
- held-out test ไม่ใช้เลือก epoch, architecture หรือ hyperparameters
- run เก่าเป็น immutable evidence และห้ามเขียนทับ

## 4. ปัญหาปัจจุบันและการตัดสินใจ

### 4.1 Time-axis mismatch

Feature tensor ปัจจุบันมีรูป `[feature_bins, time_frames, channel]` แต่โมเดล reshape CNN output โดยใช้แกน feature เป็น sequence length ทำให้ BiLSTM มีแนวโน้มอ่านกลุ่ม feature แทนเวลา

การแก้:

- คง feature contract เดิมในขั้น compatibility
- หลัง CNN ให้ transpose เป็น `[time, reduced_feature * channels]`
- เพิ่ม test ยืนยันว่า sequence length ที่เข้า BiLSTM มาจาก time axis

### 4.2 Time pooling รุนแรงเกินไป

Stage 2 ใช้ 2x2 pooling หลายครั้ง ทำให้ 128 time frames เหลือประมาณ 8 ก่อนเข้า recurrent layers

การแก้ candidate:

- Stage 1 ใช้ pooling `(2,2), (2,2), (2,1)`
- Stage 2 ใช้ pooling `(2,2), (2,2), (2,1), (2,1)`
- แกน feature ถูกลดต่อ แต่แกน time เหลือประมาณ 32 frames
- shape tests ต้องตรวจทุก block

### 4.3 Heterogeneous feature blocks

Stage 2 ต่อ MFCC, Delta, Delta2, Log-Mel และ Chroma บนแกนเดียว ทำให้ Conv2D kernel ข้าม boundary ระหว่าง feature ที่มีสเกลและความหมายต่างกัน

จะเปรียบเทียบอย่างน้อย:

1. `corrected_single_branch`: tensor เดิม + time-axis correction
2. `corrected_multi_branch`: MFCC derivatives, Log-Mel และ Chroma แยก CNN branch แล้วรวม embedding

โมเดลหลักจะเลือกจาก grouped OOF metrics ไม่ใช่ held-out Test

### 4.4 Normalization

จะเพิ่ม normalizer ที่บันทึกค่า per-feature-bin หรือ per-block จาก training partition เท่านั้น โดย Validation/Test ใช้ค่าจาก bundle ของ fold/final-refit ห้าม fit ใหม่

ต้องมี compatibility manifest ระบุ:

- normalization mode
- feature order และ block boundaries
- mean/std shape
- preprocessing version
- SHA-256 ของ normalizer

### 4.5 Augmentation และ Mixup

จะไม่ลบ augmentation เดิมทันที แต่ประเมินด้วย ablation:

- no augmentation
- waveform augmentation only
- waveform augmentation + Mixup
- candidate realistic augmentation ที่ไม่เปลี่ยน semantic cue รุนแรง

Original group assignment ต้องเกิดก่อนสร้าง derivative เสมอ

### 4.6 Accuracy target

Stage 2 เดิมมี Final Test accuracy `275/303 = 90.76%`; ค่าอย่างน้อย 97% ต้องถูกอย่างน้อย `294/303` หรือดีขึ้นสุทธิ 19 records

เป้าหมายสำหรับ development/model selection:

- grouped OOF accuracy >= 97%
- Macro-F1 >= 95%
- Balanced Accuracy >= 95%
- recall ทุกคลาส >= 90%

เป้าหมายไม่ใช่การรับประกัน ผลต้องรายงานตามจริง และห้ามเลือกโมเดลจาก Test เพื่อบังคับให้ถึง 97%

## 5. Baseline และ Ablation

### 5.1 Baselines

Stage 1:

- majority/dummy classifier
- MFCC summary + SVM
- Log-Mel Small CNN

Stage 2:

- majority/dummy classifier
- MFCC summary + SVM
- Log-Mel Small CNN
- YAMNet embedding + linear/MLP classifier

### 5.2 Ablations

- CNN only
- CNN + BiLSTM without Attention
- corrected CNN + BiLSTM + Attention
- feature block ablation
- augmentation/Mixup ablation
- normalization ablation

ทุก experiment ต้องใช้ original cohort, exclusions, group rules และ fold assignments เดียวกัน พร้อมบันทึก fold-assignment hash

## 6. GPU Runtime

Environment ที่ยืนยันแล้ว:

- WSL2 Ubuntu 22.04
- Python 3.10
- TensorFlow 2.21.0 / Keras 3
- RTX 5070 Ti
- CUDA libraries จาก `tensorflow[and-cuda]`
- TensorFlow ตรวจพบ `/GPU:0`
- RTX 5070 Ti ใช้ PTX JIT เพราะ wheel ยังไม่มี native kernel binary สำหรับ compute capability 12.0a

Trainer จะรองรับ:

- `--device auto|gpu|cpu`
- `--require-gpu`
- `--mixed-precision` แบบ explicit
- GPU memory growth
- float32 classification output/loss boundary
- environment/device manifest
- fail fast เมื่อ `--require-gpu` แต่ไม่พบ GPU

GPU เปลี่ยนความเร็ว ไม่เปลี่ยน evaluation protocol และไม่รับประกัน accuracy เพิ่มขึ้น

## 7. Feature Cache

Feature extraction ของ librosa ยังทำบน CPU จึงเพิ่ม cache เพื่อลดการคำนวณซ้ำ

Cache key ต้องประกอบด้วย:

- source SHA-256
- preprocessing version/config hash
- augmentation type/parameters/seed หากเป็น derivative
- feature dtype/shape

ข้อกำหนด:

- original feature cache ใช้ซ้ำได้หาก config hash ตรง
- augmented cache แยกตาม run/fold plan
- normalizer ยัง fit เฉพาะ fold training tensors
- cache ไม่เปลี่ยน record/group identity
- corrupted/stale cache ต้องถูก reject ไม่ใช่นำมาใช้เงียบ ๆ

## 8. Directory Design

```text
INFANT CRY/
├── file.txt
├── requirements/
│   ├── gpu-py310.txt
│   └── gpu-environment-lock.txt
├── cryinsight/
│   ├── audio/
│   │   └── features.py
│   ├── models/
│   │   ├── attention.py
│   │   ├── stage1_model.py
│   │   └── stage2_model.py
│   ├── runtime/
│   │   └── device.py
│   └── training/
│       ├── protocol.py
│       ├── artefacts.py
│       └── feature_cache.py
├── Models_dbl/
│   ├── binary/
│   │   ├── train_binary_dbl.py
│   │   └── runs/
│   ├── Main/
│   │   ├── train_main_dbl.py
│   │   └── runs/
│   └── experiments/
│       ├── README.md
│       ├── configs/
│       ├── baselines/
│       │   ├── stage1/
│       │   └── stage2/
│       ├── ablations/
│       └── runs/
├── tests/
├── Report/
│   ├── report.md
│   ├── runs/
│   ├── experiments/
│   └── assets/
├── README.md
└── Architecture.md
```

ไม่สร้าง `cryinsight/inference/audio_contract.py`, `audio/windowing.py` หรือ `audio/quality.py` ในรอบนี้ เพราะเป็นส่วนของ external-audio scope ที่ยังไม่อนุมัติ

## 9. Run Artefacts

รูปแบบ Stage run ยังคง:

```text
runs/<run_id>/
├── fold_1/ ... fold_5/
├── oof_predictions.csv
├── oof_metrics.json
├── final_test_predictions.csv
├── final_test_metrics.json
├── environment.json
├── verification.json
└── best_model/
    ├── best_model_<stage>_dbl.keras
    ├── norm_stats_<stage>_dbl.npy
    ├── labels_<stage>_dbl.json
    ├── preprocessing_config.json
    └── deployment_manifest.json
```

Experiment run เพิ่ม:

```text
Models_dbl/experiments/runs/<experiment_id>/
├── protocol.json
├── environment.json
├── shared_fold_assignments.csv
├── comparison_metrics.csv
├── comparison.md
├── stage1/
└── stage2/
```

## 10. Publication Scope

### แก้/รายงานในรอบนี้

- rename claims จาก clinical emotion recognition เป็น infant cry/state classification ตามบริบท
- baseline และ ablation evidence
- multiple seeds
- end-to-end cascade metrics
- ROC/PR curves
- environment reproducibility
- GPU/device/runtime evidence

### คงเป็นข้อจำกัด

- subject/session identity ยังไม่ยืนยันครบ
- Stage 1 source confounding
- ยังไม่มี external validation
- labels เป็น operational dataset labels
- dataset citation, annotation, ethics, consent และ license ต้องอ้างแหล่งต้นฉบับ

ประเด็น legacy unpublished model ไม่ถูกจัดเป็น blocker ตามคำสั่งผู้ใช้ แต่ผลใหม่ยังอธิบายตาม evaluation scope ที่ตรวจสอบได้และไม่อ้างเป็น external validation

## 11. Compatibility and Error Handling

ก่อนแก้ architecture ต้องตรวจ:

- current unit tests บน WSL
- model build ทั้งสอง Stage บน GPU
- one forward/backward mini-batch
- AttentionLayer Keras 3 serialization
- `.keras` save/load
- optimizer/loss compatibility
- NumPy 2.2.6 feature extraction compatibility

Trainer ต้องหยุดพร้อมข้อความชัดเจนเมื่อ:

- require GPU แต่ไม่พบ GPU
- model/preprocessing/normalizer shape ไม่ตรง
- cache hash/config ไม่ตรง
- non-finite tensors/probabilities
- OOF coverage ไม่ครบหรือซ้ำ
- augmented sample ปรากฏใน Validation/OOF

## 12. Testing

เพิ่มหรือแก้ tests สำหรับ:

- GPU device selection/fail-fast
- mixed-precision output dtype
- Stage 1/Stage 2 time-axis sequence shape
- pooling shape
- per-block normalizer leakage boundary
- feature cache key/invalidation
- Keras 3 save/load
- baseline shared fold assignments
- ablation protocol
- end-to-end cascade aggregation
- run verification/environment manifest

Smoke test อนุญาตเฉพาะ tiny synthetic/mini-batch run ไม่ใช่ full fold training

## 13. Documentation

หลัง implementation ต้องอัปเดต:

- `README.md`: Python 3.10 + WSL2 + TensorFlow GPU, interpreter, commands, directory tree, Baseline/Ablation และ limitations
- `Architecture.md`: corrected time axis, pooling, candidate multi-branch, GPU runtime, cache และ evaluation design
- `Report/report.md`: link ไป baseline/ablation report
- `Report/experiments/baseline_report.md`
- `Report/experiments/ablation_report.md`
- `file.txt`: คำสั่งใช้งาน WSL/Ubuntu/VS Code/GPU

`README_OLD.md` เป็น legacy reference และจะไม่แก้

## 14. Implementation Boundaries

- ไม่ลบหรือย้าย Dataset
- ไม่ลบ run เก่า
- ไม่เขียนทับ immutable run
- ไม่เริ่ม full training
- ไม่สร้าง metric จำลอง
- ไม่ implement external-audio scope
- ไม่เลือก best fold เป็น final model
- ไม่ปรับ Test ซ้ำเพื่อไล่เป้าหมาย 97%

## 15. Acceptance Criteria

ก่อนส่งคำสั่ง Full Training ต้องมีหลักฐานว่า:

1. tests ผ่านใน WSL GPU environment
2. TensorFlow พบ GPU และ mini-batch ใช้ `/GPU:0`
3. Stage 1/2 sequence axis เป็น Time ตาม test
4. Keras 3 save/load ผ่าน
5. baseline/ablation ใช้ shared fold assignments
6. preprocessing/normalizer/cache manifests ตรวจ hash ได้
7. validation/OOF leakage assertions ยังผ่าน
8. README และ Architecture ตรงกับโค้ด
9. external-audio feature ไม่มีการเพิ่มโดยไม่ได้อนุมัติ
10. ไม่มี full training ถูกเริ่มโดย Codex
