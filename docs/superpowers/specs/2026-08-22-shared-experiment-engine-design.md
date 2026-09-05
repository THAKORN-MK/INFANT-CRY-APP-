# Shared Experiment Engine Design

วันที่: 2026-08-22

สถานะ: ผู้ใช้อนุมัติสเปกและ Inline Execution แล้ว; ติดตามการเก็บงานและหลักฐานตรวจรับที่ [Inline Execution Status](../plans/2026-09-03-inline-execution-status.md)

ขอบเขตโครงการ: `D:/INFANT CRY`

Baseline อ้างอิง (การเทรน/การทดลองครั้งที่ 2): `20260821T164332Z_490383ff`

## 1. วัตถุประสงค์

สร้าง Shared Experiment Engine ที่ทำให้ Baseline, Ablation และ Proposed Model ใช้ข้อมูล กลุ่ม Fold การประเมิน และรูปแบบผลลัพธ์ชุดเดียวกัน เพื่อหาหลักฐานว่าการเปลี่ยน Feature, Architecture, Augmentation หรือ Training Strategy ส่วนใดช่วย Stage 2 จริง ก่อนนำค่าที่ชนะไปสร้างการเทรนครั้งที่ 3

ระบบต้อง:

1. ใช้ `fold_assignments.csv` เดิมจากการทดลองครั้งที่ 2 โดยไม่สุ่มแบ่งใหม่
2. ฝึกและประเมิน Candidate ด้วย grouped 5-fold OOF เท่านั้น
3. ไม่อ่าน `data_set_dbl_split/test` และไม่ใช้ Final Test เลือก Candidate
4. รองรับทั้ง scikit-learn และ TensorFlow/Keras ผ่านสัญญาการทำงานเดียวกัน
5. เปรียบเทียบ Baseline, Ablation และโมเดลหลักด้วย Metric และ Artefact รูปแบบเดียวกัน
6. รัน Candidate แบบเป็นรอบ เพื่อลดเวลาและไม่ทดลองทุก Combination โดยไม่จำเป็น
7. ยืนยัน Candidate ที่ดีที่สุดด้วยหลาย Random Seeds
8. สร้างผลที่ตรวจสอบย้อนกลับได้และไม่เขียนทับ Run ที่เสร็จแล้ว
9. ยังไม่แก้ค่า Final ใน `Models_dbl/Main/train_main_dbl.py` จนกว่าจะได้ผู้ชนะจาก Experiment

เป้าหมาย Accuracy 97% เป็นเป้าหมายเชิงผลลัพธ์ ไม่ใช่สิ่งที่ Engine รับประกัน ระบบต้องรายงานค่าจริงแม้ต่ำกว่าเป้าหมาย

## 2. ขอบเขต

### 2.1 อยู่ในขอบเขต

- Shared grouped 5-fold experiment runner
- Candidate registry และ Config ที่ตรวจสอบได้
- Feature views สำหรับ Classical และ Neural models
- Stage 1/Stage 2 Baselines ที่มีอยู่ใน Registry
- Stage 2 Architecture, Feature, Augmentation, Normalization และ Loss Ablations
- Train-only normalization และ augmentation
- GPU/runtime checks สำหรับ Neural candidates
- Native-Linux checkpoint staging สำหรับ Keras บน WSL
- OOF prediction aggregation, metrics, leaderboard และ selection manifest
- Seed 42 สำหรับ screening และ seeds 42/123/2026 สำหรับ confirmation
- Resume เฉพาะ Run ที่ยังไม่เสร็จ โดยไม่เขียนทับ Fold ที่ยืนยัน complete แล้ว
- Tests และเอกสารที่เกี่ยวข้อง

### 2.2 ไม่อยู่ในขอบเขต

- ไม่เปลี่ยน `data_set_dbl_split/train` หรือ `data_set_dbl_split/test`
- ไม่เปิด Final Test เพื่อจัดอันดับ Candidate
- ไม่ลบหรือแก้ Run `20260821T164332Z_490383ff`
- ไม่เปลี่ยน Webapp หรือ Deployment bundle ในรอบนี้
- ไม่เพิ่ม External-audio preprocessing contract
- ไม่แก้ Subject/session identity ที่ Dataset ไม่มี Metadata ยืนยัน
- ไม่เริ่ม Full Experiment training โดย Codex
- ไม่โปรโมต Candidate เข้า `train_main_dbl.py` อัตโนมัติ
- ไม่สร้างการเทรนครั้งที่ 3 จนกว่าผู้ใช้จะอนุมัติผล Experiment และ Promotion Spec

## 3. สถานะปัจจุบันที่ต้องรักษา

การทดลองครั้งที่ 2 ใช้ Run ID `20260821T164332Z_490383ff` ร่วมกันระหว่าง Stage 1 และ Stage 2 และมีสถานะ `complete`

Stage 2 Baseline อ้างอิง:

- OOF support: 1,045 original records
- OOF Accuracy: 91.10%
- OOF Macro F1: 88.78%
- OOF Balanced Accuracy: 88.72%
- Final Test support: 303 original records
- Final Test Accuracy: 92.41%
- Final Test Macro F1: 91.02%

ค่า Final Test ใช้บรรยายผลของการทดลองครั้งที่ 2 เท่านั้น ห้ามนำไปใช้เลือก Candidate ของ Shared Experiment Engine

Engine ต้องบันทึก Baseline อ้างอิงเป็น `reference_run.json` โดยใช้ค่า OOF จาก Artefact เดิม พร้อม SHA-256 ของ `verification.json`, `fold_assignments.csv`, `oof_predictions.csv` และ `oof_metrics.json` ห้ามคัดลอก Final Test metrics เข้า Leaderboard ที่ใช้เลือกโมเดล

## 4. แนวทางที่เลือก

ใช้ Shared Engine แบบ Config-driven และ Adapter-based

- Config กำหนด Candidate, Stage, Feature view, Model, Augmentation, Normalization, Loss, Seed และ Resource requirements
- Adapter แปลง Candidate แต่ละชนิดให้ใช้ lifecycle เดียวกัน
- Runner รับผิดชอบ Fold, provenance, artefact และ failure handling
- Selection module รับเฉพาะ verified OOF metrics
- Definition scripts เดิมยังคงเป็นสคริปต์ขนาดเล็ก เพราะทำหน้าที่ลงทะเบียนและตรวจนิยาม ไม่ทำซ้ำ Training loop

ไม่เลือกวิธีคัดลอก Training loop ไปไว้ในทุก Baseline script เพราะจะทำให้ Fold handling, leakage checks, metrics และ artefact แตกต่างกันและตรวจสอบงานวิจัยยาก

## 5. สถาปัตยกรรม

### 5.1 Core modules ใหม่

```text
cryinsight/
└── experiments/
    ├── __init__.py
    ├── contracts.py
    ├── registry.py
    ├── fold_data.py
    ├── feature_views.py
    ├── classical.py
    ├── neural.py
    ├── runner.py
    ├── selection.py
    └── reporting.py
```

หน้าที่ของแต่ละ Module:

- `contracts.py`: Dataclass/Protocol สำหรับ Candidate, Fold request, Fold result และ Verification result
- `registry.py`: Source of truth ของ Candidate definitions และ validation ของ Config
- `fold_data.py`: โหลด record ตาม shared assignment, ตรวจ Group/Fold และสร้าง train/validation view
- `feature_views.py`: สร้าง labels-only, MFCC summary, Log-Mel, all feature blocks, block subsets และ YAMNet embedding
- `classical.py`: Adapter สำหรับ Dummy/SVM/Linear/MLP แบบ scikit-learn
- `neural.py`: Adapter สำหรับ Small CNN, CNN/BiLSTM/Attention และ Multi-branch TensorFlow models
- `runner.py`: Lifecycle prepare → train folds → aggregate OOF → verify → summarize
- `selection.py`: Eligibility, ranking, repeated-seed aggregation และ promotion recommendation
- `reporting.py`: Leaderboard CSV/Markdown, comparison report และกราฟที่มาจาก OOF เท่านั้น

`cryinsight/training/experiments.py` เดิมจะเปลี่ยนเป็น compatibility re-export หรือ thin wrapper ไปยัง `cryinsight.experiments` เพื่อไม่ให้มี Registry สองชุด

### 5.2 Entry point

`Models_dbl/experiments/run_experiments.py` เป็น Entry point เดียวและรองรับ:

```text
--audit-only     ตรวจ Config/Registry โดยไม่สร้าง Run
--prepare-only   สร้าง immutable protocol และ shared assignment snapshot
--train          ฝึก Candidate ตาม Config
--resume         ทำต่อเฉพาะ Candidate/Seed/Fold ที่ยังไม่ complete
--summarize      รวม verified OOF results และสร้าง Leaderboard
```

Training command จะรับ `--pipeline-run-id 20260821T164332Z_490383ff` และ `--config <path>` อย่างชัดเจน หาก Run ID ของ Stage 1/Stage 2 ไม่ตรงกันหรือ Verification ไม่ complete ต้องหยุด

### 5.3 Candidate adapter contract

Adapter ทุกชนิดต้องทำงานตามลำดับเดียวกัน:

1. รับ Candidate config, Fold data และ Seed
2. Fit preprocessing/normalizer จาก Fold training originals เท่านั้น
3. สร้าง augmentation เฉพาะ Fold training เมื่อ Config อนุญาต
4. Fit model
5. Predict probability สำหรับ original validation records เท่านั้น
6. บันทึก model, preprocessing, predictions, metrics และ provenance
7. คืน `FoldResult` ที่ Runner ตรวจสอบได้

Model-specific logic อยู่ใน Adapter แต่การแบ่ง Fold, OOF aggregation, metrics, path naming และ verification อยู่ใน Shared Engine เท่านั้น

## 6. Data และ Leakage Contract

### 6.1 แหล่งข้อมูล Development

- Stage 1 ใช้ original development cohort และ `fold_assignments.csv` จาก `Models_dbl/binary/runs/20260821T164332Z_490383ff`
- Stage 2 ใช้ original development cohort และ `fold_assignments.csv` จาก `Models_dbl/Main/runs/20260821T164332Z_490383ff`
- Assignment snapshot ของ Experiment ต้องมี SHA-256 ตรงกับ Reference run
- Record ID ต้องครบ ไม่ซ้ำ และทุก Group อยู่ใน Validation fold เดียว
- Validation ของแต่ละ Fold มี original records เท่านั้น

### 6.2 Final Test lock

Experiment CLI จะไม่มี Argument สำหรับ Test dataset และไม่มีขั้น Final Test evaluation

Runner ต้องปฏิเสธ:

- Path ที่ resolve เข้า `data_set_dbl_split/test`
- Config field ที่ชื่อ `test`, `heldout`, `final_test` หรือ Selection metric ที่มีคำเหล่านี้
- Artefact จาก Test ที่ถูกส่งเข้ากระบวนการ ranking

ผลลัพธ์ Experiment จึงเป็น Development/OOF evidence เท่านั้น

### 6.3 Augmentation

- Group/Fold assignment เกิดก่อน augmentation เสมอ
- Augmented sample สืบทอด `source_record_id` และ `group_id`
- Augmented sample อยู่ใน training side ของ Fold ต้นทางเท่านั้น
- OOF support นับ original validation records เท่านั้น
- Manifest ต้องบันทึก augmentation type, parameters, seed และ source record

### 6.4 Normalization

- Fit statistics จาก Fold training tensors เท่านั้น
- Validation ใช้ statistics ของ Fold นั้นโดยห้าม fit ใหม่
- Normalizer artefact ต้องระบุ shape, feature boundaries, dtype, config hash และ SHA-256

## 7. Feature Views

Engine รองรับ Feature views ที่มีขอบเขตชัดเจน:

| ID | Input | ใช้กับ |
|---|---|---|
| `labels_only` | Labels เท่านั้น | Majority baseline |
| `mfcc_summary` | Mean/std และ summary statistics ของ MFCC | SVM baseline |
| `log_mel` | Log-Mel tensor | Small CNN baseline |
| `yamnet_embedding` | Frozen YAMNet embeddings | Linear/MLP baseline |
| `all_blocks` | MFCC + Delta + Delta2 + Log-Mel + Chroma | Proposed/Ablation neural models |
| `feature_block_subset` | ชุดย่อยที่ระบุใน Config | Feature ablation |
| `multi_branch_blocks` | MFCC derivatives, Log-Mel และ Chroma แยก branch | Multi-branch candidate |

Feature cache key ต้องรวม source audio SHA-256, preprocessing config hash, feature-view ID, dtype และ augmentation identity หากเป็น derivative หาก Key หรือ Shape ไม่ตรงต้องคำนวณใหม่หรือหยุด ห้ามใช้ cache ผิดชุดแบบเงียบ ๆ

## 8. Candidate Matrix และลำดับการทดลอง

เพื่อควบคุมเวลา GPU จะไม่รัน Cartesian product ของทุกค่า แต่แบ่งเป็น Waves

### 8.1 Reference

- `stage2_reference_run2`: นำเข้าเฉพาะ verified OOF metrics จาก Run ครั้งที่ 2 ไม่ฝึกซ้ำ และไม่ใช้ Final Test metrics ในการจัดอันดับ

### 8.2 Wave A: Baseline และ Architecture screening

ใช้ Seed 42 และ shared 5 folds:

- `stage2_majority`
- `stage2_mfcc_svm`
- `stage2_logmel_small_cnn`
- `stage2_yamnet_linear`
- `stage2_yamnet_mlp`
- `stage2_cnn_only`
- `stage2_cnn_bilstm`
- `stage2_corrected_attention`
- `stage2_multi_branch_attention`

วัตถุประสงค์คือวัดความยากของ Dataset และเลือก Neural architecture ที่มีหลักฐานดีที่สุด ไม่ใช่บังคับให้ Baseline ชนะ Proposed model

### 8.3 Wave B: Focused ablations

ใช้ Architecture ที่ดีที่สุดจาก Wave A, Seed 42 และ shared 5 folds

Feature:

- all blocks
- without Chroma
- without Log-Mel
- MFCC/Delta/Delta2 only

Augmentation:

- none
- waveform augmentation only
- waveform augmentation + Mixup

Normalization/Loss:

- current per-feature normalization + categorical cross-entropy
- class-balanced categorical cross-entropy
- focal loss ที่กำหนด gamma/alpha ใน Config

Wave B ใช้ one-factor-at-a-time รอบ Anchor ที่ชนะ ห้ามสร้างทุก Combination ข้ามสามกลุ่มโดยอัตโนมัติ Candidate ที่ชนะในแต่ละกลุ่มจึงค่อยรวมเป็น Candidate เดียวสำหรับ Wave C

### 8.4 Wave C: Repeated-seed confirmation

เลือกไม่เกิน 2 Candidates จาก Wave A/B แล้วรัน seeds:

- 42
- 123
- 2026

ทุก Seed ใช้ record/group/fold assignments เดิม เปลี่ยนเฉพาะ stochastic initialization, batch order และ train-only augmentation randomness

### 8.5 การเทรนครั้งที่ 3

Shared Experiment Engine ไม่สร้าง Final model ของการเทรนครั้งที่ 3

หลัง Wave C:

1. สร้าง `selection.json` และ `promotion_recommendation.md`
2. ผู้ใช้ตรวจและอนุมัติ Candidate
3. จัดทำ Promotion Spec แยกต่างหากสำหรับแก้ค่า Final ใน `train_main_dbl.py`
4. การเทรนครั้งที่ 3 จึงเริ่มด้วย Config ที่ Freeze แล้ว

## 9. Metrics และ Selection Rules

### 9.1 Eligibility ก่อนจัดอันดับ

Candidate/Seed จะเข้าสู่ Leaderboard เมื่อ:

- Fold 1-5 complete
- original OOF record ปรากฏ exactly once
- OOF record IDs และ support ตรงกับ shared assignments
- ไม่มี augmented record ใน Validation/OOF
- probability ทุกค่า finite และอยู่ในช่วง `[0, 1]`
- ผลรวม probability ต่อ record ต่างจาก 1 ไม่เกิน `1e-5`
- model/config/fold/prediction hashes ครบ
- `verification.json` มีสถานะ `complete`

Candidate ที่ไม่ผ่านจะแสดงใน Failure table แต่ไม่ถูกจัดอันดับ

ก่อนส่ง probability เข้า scikit-learn metrics ให้คำนวณเป็น float64 และ normalize ด้วยผลรวมหาก deviation ไม่เกิน `1e-5` พร้อมบันทึก `max_probability_sum_deviation` หากเกินค่าดังกล่าวต้อง Fail วิธีนี้ป้องกัน Warning เรื่อง probability sum โดยไม่ปกปิด model output ที่ผิดสัญญา

### 9.2 Metric หลัก

- Primary: grouped OOF Macro F1
- Secondary: grouped OOF Balanced Accuracy
- Safety metric: minimum per-class recall
- Diagnostic: Accuracy, weighted F1, log loss, Brier score, ECE, per-class precision/recall/F1 และ ROC-AUC เมื่อคำนวณได้

Accuracy ไม่เป็น Metric เดียวสำหรับเลือกโมเดล เพราะ Stage 2 มี Class imbalance

### 9.3 Wave A/B ranking

จัดอันดับด้วย OOF Macro F1 ของ Seed 42 Candidate ที่ต่างจากอันดับสูงสุดไม่เกิน 0.005 ถือว่าใกล้เคียงกันและใช้ minimum per-class recall เป็นตัวตัดสิน หากยังต่างกันไม่เกิน 0.005 ให้เลือกโมเดลที่มี Parameter count ต่ำกว่า จากนั้นใช้ Candidate ID เรียงตัวอักษรเพื่อให้ผล deterministic

### 9.4 Wave C ranking

จัดอันดับด้วยค่าเฉลี่ย OOF Macro F1 ของสาม Seeds และรายงาน mean, standard deviation, minimum และ maximum

หากค่าเฉลี่ยต่างกันไม่เกิน 0.005 ใช้ตามลำดับ:

1. ค่าเฉลี่ย minimum per-class recall สูงกว่า
2. OOF Macro F1 standard deviation ต่ำกว่า
3. Parameter count ต่ำกว่า
4. Candidate ID เรียงตัวอักษร

### 9.5 Promotion recommendation

Candidate จะได้รับสถานะ `recommended_for_training_3` เมื่อ:

- ผ่าน Wave C ครบสาม Seeds
- mean OOF Macro F1 สูงกว่า Reference อย่างน้อย 0.01
- mean OOF Balanced Accuracy ไม่ต่ำกว่า Reference
- mean minimum per-class recall ไม่ต่ำกว่า Reference เกิน 0.02
- ไม่มี Leakage/Verification failure

หากไม่มี Candidate ผ่าน ให้รายงาน `no_promotion_recommended` และรักษา `train_main_dbl.py` เดิม ไม่ลดเกณฑ์เพื่อบังคับให้มีผู้ชนะ

## 10. Run Identity และ Directory Design

### 10.1 Identity

- `pipeline_run_id`: Run คู่ Stage 1/Stage 2 ที่เป็นข้อมูลอ้างอิง เช่น `20260821T164332Z_490383ff`
- `experiment_run_id`: Run ของ Shared Engine รูปแบบ `<pipeline_run_id>__exp_<UTC timestamp>_<8 hex>`
- `candidate_id`: ชื่อนิยาม Candidate ที่คงที่
- `seed`: Seed ของการฝึก
- `fold`: 1-5

ตัวอย่าง:

```text
20260821T164332Z_490383ff__exp_20260822T120000Z_a1b2c3d4
```

ชื่อและ Metadata ทำให้รู้ว่า Experiment ใดอ้างอิงการเทรนครั้งที่ 2 แม้สร้างคนละเวลา

### 10.2 Directory tree

```text
Models_dbl/experiments/
├── README.md
├── run_experiments.py
├── configs/
│   ├── stage1_baselines.json
│   ├── stage2_wave_a.json
│   ├── stage2_wave_b_features.json
│   ├── stage2_wave_b_augmentation.json
│   ├── stage2_wave_b_loss.json
│   └── stage2_wave_c.json
├── baselines/
│   ├── stage1/
│   └── stage2/
├── ablations/
└── runs/
    └── <experiment_run_id>/
        ├── protocol.json
        ├── reference_run.json
        ├── shared_fold_assignments.csv
        ├── candidate_matrix.json
        ├── environment.json
        ├── leaderboard.csv
        ├── leaderboard.md
        ├── selection.json
        ├── promotion_recommendation.md
        ├── verification.json
        └── candidates/
            └── <candidate_id>/
                └── seed_<seed>/
                    ├── config.json
                    ├── environment.json
                    ├── fold_1/
                    ├── fold_2/
                    ├── fold_3/
                    ├── fold_4/
                    ├── fold_5/
                    ├── oof_predictions.csv
                    ├── oof_metrics.json
                    ├── seed_summary.json
                    └── verification.json
```

Neural fold เก็บ `.keras`, history, normalizer, predictions, metrics, checkpoint publication และ manifest ส่วน Classical fold เก็บ estimator/preprocessor ด้วยรูปแบบที่กำหนดใน manifest และ predictions/metrics ชุดเดียวกัน

Experiment ไม่มี `best_model/` เพราะ Final-refit model จะเกิดเฉพาะการเทรนครั้งที่ 3 หลัง Promotion approval

## 11. Run State, Immutability และ Resume

สถานะ Run:

```text
prepared → running → complete
                   ↘ failed
```

- `prepared`: protocol และ assignment snapshot พร้อม แต่ยังไม่ฝึก
- `running`: อนุญาตให้เพิ่ม Artefact ของงานที่ยังไม่เสร็จ
- `complete`: ตรวจครบและห้ามแก้ Artefact
- `failed`: เก็บ Failure manifest และอนุญาต `--resume` เฉพาะงานที่ไม่ complete

Rules:

- ห้ามเขียนทับ Candidate/Seed/Fold ที่มี verified complete marker
- `--resume` ต้องใช้ Config hash, assignment hash และ environment contract เดิม
- หาก Hash ต่างต้องสร้าง `experiment_run_id` ใหม่
- Keras checkpoints ใช้ `cryinsight.training.checkpoint_staging` บน Native Linux แล้ว publish แบบ verified copy ไป `/mnt/d`
- การหยุดกลางทางไม่ทำให้ Fold ที่ complete สูญหาย
- Final `verification.json` บันทึกจำนวน expected/completed/failed jobs และ Artefact hashes

## 12. Error Handling

Engine ต้องหยุด Candidate หรือ Run พร้อมสาเหตุชัดเจนเมื่อ:

- Reference pipeline run ไม่มีหรือ Verification ไม่ complete
- Stage 1/Stage 2 pipeline Run ID ไม่ตรงกัน
- Fold assignment hash ไม่ตรง Reference
- Record/group/label ไม่ตรง shared assignment
- Augmented record ปรากฏใน Validation/OOF
- Feature/cache/preprocessing shape หรือ hash ไม่ตรง
- Probability non-finite, นอกช่วง หรือ sum deviation เกิน tolerance
- OOF coverage ขาดหรือซ้ำ
- Config/selection metric อ้างถึง Test/Held-out
- `--require-gpu` แต่ TensorFlow ไม่พบ GPU
- Checkpoint publish/hash verification ล้มเหลว
- พยายามแก้ Run ที่ complete

Failure ของ Candidate หนึ่งตัวต้องถูกบันทึกและไม่ถูกนำไป Ranking ส่วน Runner สามารถไป Candidate ถัดไปได้ตาม Config เว้นแต่เป็น Protocol-level failure เช่น Fold hash หรือ Test leakage ซึ่งต้องหยุดทั้ง Run

## 13. Testing Strategy

### 13.1 Unit tests

- Registry/config schema validation
- Candidate adapter contract
- Reference run และ assignment hash validation
- Fold train/validation isolation
- Train-only normalization/augmentation
- Feature-view shapes และ cache keys
- Probability validation/float64 metric normalization
- Ranking/tie-break/promotion rules
- State transition, immutability และ resume
- Rejection ของ Test path/metric/config fields

### 13.2 Integration tests

- Tiny synthetic 5-fold run สำหรับ Majority/SVM
- Tiny synthetic 5-fold run สำหรับ Small CNN หนึ่ง Epoch
- Neural checkpoint staging/save/load บน WSL
- OOF aggregation exactly once
- Interrupted run แล้ว resume โดยไม่เขียนทับ Fold complete
- Mixed valid/failed Candidate leaderboard

### 13.3 Regression tests

- Registry เดิมยัง Audit ได้
- Existing Stage 1/Stage 2 trainers ยังทำงานตาม Contract เดิม
- Run `20260821T164332Z_490383ff` ไม่ถูกแก้
- Full test suite เดิมต้องผ่าน

Test ห้ามทำ Full Dataset training

## 14. Documentation และ Reports

หลัง Implementation ต้องอัปเดต:

- `Models_dbl/experiments/README.md`: คำสั่ง Audit, Prepare, Train, Resume และ Summarize
- `README.md`: ลิงก์ Experiment workflow และแยก Training run ออกจาก Experiment run
- `Architecture.md`: Shared Engine, OOF-only selection และ Promotion flow
- `Report/report.md`: ลิงก์หน้า Experiment hub
- `Report/experiments/report.md`: หน้า Menu ของ Experiment reports
- `file.txt`: คำสั่ง WSL สำหรับรันแต่ละ Wave

แต่ละ Experiment run สร้าง Report ที่ `Report/experiments/<experiment_run_id>.md` จาก verified artefacts เท่านั้น และต้องระบุว่า Final Test ไม่ได้ใช้จัดอันดับ

`README_OLD.md` คงเป็น Legacy reference และไม่แก้

## 15. การเปลี่ยนไฟล์เดิม

ไฟล์ที่จะเปลี่ยนเมื่อเริ่ม Implementation:

- `Models_dbl/experiments/run_experiments.py`
- `Models_dbl/experiments/script_support.py`
- `Models_dbl/experiments/configs/*.json`
- `Models_dbl/experiments/baselines/**/*.py`
- `Models_dbl/experiments/ablations/*.py`
- `cryinsight/training/experiments.py` เพื่อ Compatibility
- Tests และเอกสารในข้อ 14

ไฟล์ที่จะสร้าง:

- `cryinsight/experiments/*.py`
- Config แยกตาม Wave
- Tests ของ Shared Engine
- Experiment report hub

ไฟล์ที่ยังไม่แก้ใน Implementation รอบนี้:

- `Models_dbl/Main/train_main_dbl.py`
- `Models_dbl/binary/train_binary_dbl.py`
- Webapp/Inference code
- Dataset และ completed run artefacts

## 16. Acceptance Criteria

Implementation พร้อมให้ผู้ใช้เริ่ม Wave A เมื่อ:

1. Audit/Prepare/Train/Resume/Summarize lifecycle ผ่าน Tests
2. Shared assignments ตรงกับ Run `20260821T164332Z_490383ff` และมี Hash evidence
3. Test dataset ไม่สามารถเข้าสู่ Training/Ranking path
4. Classical และ Neural adapter ผ่าน Tiny 5-fold integration tests
5. OOF coverage, probability และ leakage verification ผ่าน
6. Candidate ที่ไม่ complete ไม่เข้าสู่ Leaderboard
7. Ranking และ Promotion rules ให้ผล deterministic
8. Completed Experiment run เขียนทับไม่ได้ และ Failed run resume ได้อย่างปลอดภัย
9. Existing trainers และ Test suite ไม่ถดถอย
10. README, Architecture, Experiment README, Report hub และ `file.txt` ตรงกับคำสั่งจริง
11. Codex ยังไม่ได้เริ่ม Full Experiment training
12. `train_main_dbl.py` ยังไม่เปลี่ยนจนกว่าจะมี Wave C winner และผู้ใช้อนุมัติ Promotion

## 17. ผลลัพธ์ที่คาดหวัง

เมื่อทำตาม Design นี้ จะตอบได้ด้วยหลักฐานว่า:

- โมเดลปัจจุบันดีกว่า Majority/Classical/Transfer baselines เพียงใด
- CNN, BiLSTM และ Attention แต่ละส่วนช่วยหรือไม่
- Chroma, Log-Mel และ MFCC derivatives ช่วยคลาสใด
- Augmentation และ Mixup ช่วย Generalization หรือเพิ่มเพียงจำนวนข้อมูล
- Candidate ใดเสถียรข้าม Seeds
- มีเหตุผลเพียงพอหรือไม่ที่จะเปลี่ยน `train_main_dbl.py` สำหรับการเทรนครั้งที่ 3

หากไม่มี Candidate ที่ดีขึ้นตาม Promotion criteria ผลลัพธ์ที่ถูกต้องคือเก็บโมเดลเดิมและรายงานว่า Experiment ไม่สนับสนุนการเปลี่ยน ไม่ใช่ปรับ Test หรือเลือก Fold เพื่อให้ได้ 97%
