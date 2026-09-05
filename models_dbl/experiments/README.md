# Shared Experiment Engine

ระบบนี้ใช้เปรียบเทียบ Baseline, Ablation และ Candidate architecture ภายใต้ข้อมูลและ grouped 5-fold assignments ชุดเดียวกัน โดยใช้ **corrected grouped OOF เท่านั้น** สำหรับการจัดอันดับและ Promotion Gate

Final Test ไม่ถูกเปิดใช้ระหว่าง Wave A, Wave B หรือ Wave C และ CLI ไม่มีตัวเลือกสำหรับส่ง Test dataset เข้ามา ห้ามนำคะแนน Test มาใช้เลือก Candidate หรือปรับ Config ภายนอก Engine เช่นกัน

## Run ID สองชนิด

- **Pipeline Run ID** เช่น `20260821T164332Z_490383ff` คือ Run อ้างอิงที่ Stage 1 และ Stage 2 เทรนเสร็จสมบูรณ์แล้ว ระบบนำ fold assignments และ OOF evidence จาก Run นี้มาตรึงไว้
- **Experiment Run ID** มีรูปแบบ `<pipeline_run_id>__exp_<UTC>_<8-hex>` ใช้รวม Candidate/Seed/Fold ของการเปรียบเทียบหนึ่งครั้ง

Experiment Run ใหม่ทุกตัวจึงระบุได้ทันทีว่าอ้างอิง Pipeline Run ใด และไม่ปะปนกับ Experiment Run อื่น

## ลำดับการทดลอง

```text
Pipeline run ที่ verification=complete
                ↓
Wave A — screening baselines/architectures, seed 42
                ↓
Wave B_features → B_augmentation → B_loss
                ↓
Wave C — ยืนยัน Candidate 2 อันดับแรกด้วย seeds 42/123/2026
                ↓
Promotion Gate
                ↓
ขออนุมัติแยกก่อนแก้ trainer หลักหรือเริ่ม Training 3
```

Wave B_features เลือก Neural architecture ของ Stage 2 ที่อันดับสูงสุดในกลุ่มที่รองรับ feature blocks ครบ โดยไม่เปลี่ยนอันดับ Baseline ใน Leaderboard; Majority/SVM/YAMNet หรือ Log-Mel-only ไม่ถูกนำมาใช้เป็น anchor ของการตัด Chroma/Log-Mel หากไม่มี architecture ที่เข้าเงื่อนไข ระบบหยุดและไม่เดาแทน

Wave B ขั้นถัดไปใช้ anchor ที่ชนะพร้อม one-factor variants และบันทึก parent/อันดับที่เลือกไว้ ส่วน Wave C ใช้ Candidate ที่เข้าเงื่อนไขไม่เกินสองอันดับแรกจาก B_loss ทุกขั้นต้องเป็น Pipeline Run เดียวกันและมี parent ตรงลำดับ `A → B_features → B_augmentation → B_loss → C` พร้อมตรวจ hashes Candidate ที่ล้มเหลวหรือ verification ไม่ครบไม่เข้าสู่ Leaderboard

## ตรวจ Wave A โดยไม่สร้าง Run

```bash
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
cd "/mnt/d/INFANT CRY"
python Models_dbl/experiments/run_experiments.py --audit-only --config Models_dbl/experiments/configs/stage2_wave_a.json
```

Audit ตรวจ config, Registry, การพบ dependencies และ YAMNet archive โดยไม่สร้างโมเดล ไม่สร้าง Run และไม่เริ่ม training การพบ TensorFlow ไม่ได้ยืนยันว่า CUDA/GPU ใช้งานได้ ต้องผ่าน GPU/checkpoint smoke ใน environment ที่จะใช้จริงด้วย

## เตรียม Wave A โดยไม่เทรน

```bash
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
cd "/mnt/d/INFANT CRY"
python Models_dbl/experiments/run_experiments.py --prepare-only --pipeline-run-id 20260821T164332Z_490383ff --config Models_dbl/experiments/configs/stage2_wave_a.json
```

คำสั่งนี้ตรวจ reference hashes, ตรวจ Train records, ตรึง Candidate matrix และสร้าง Experiment Run สถานะ `prepared` เท่านั้น หลังรันต้องตรวจ `protocol.json`, `resolved_config.json`, `candidate_matrix.json`, `shared_fold_assignments.csv` และ `state.json` ก่อน

ค่า device, require-GPU, mixed precision, cache และ failure policy ถูกตรึงในขั้น Prepare หากต้องการบังคับ GPU ให้เพิ่ม `--device gpu --require-gpu --mixed-precision` ในคำสั่ง Prepare การไม่ใส่ flags ตอน Train/Resume หมายถึงใช้ค่าที่ตรึงไว้ ไม่ใช่เปลี่ยนกลับเป็น default และถ้าระบุ flags ที่ขัดกับค่าเดิมระบบจะปฏิเสธ

ยังไม่แสดงคำสั่งเริ่ม Training เป็นคำสั่งพร้อมคัดลอกในเอกสารนี้ จนกว่าจะตรวจ Prepared Run และได้รับอนุมัติเริ่ม Wave A แยกต่างหาก รูปแบบการใช้แต่ละโหมดคือ:

| โหมด | ข้อมูลที่ต้องระบุ | หน้าที่ |
|---|---|---|
| `--train` | `--pipeline-run-id`, `--config`, `--experiment-run-id` | ตรวจ Config ต้นฉบับทั้งชุดแล้วใช้ Prepared Run เดิม |
| `--resume` | `--experiment-run-id` | รันงานที่ยังไม่ complete ใน Run เดิม โดยรักษา attempt เก่า |
| `--summarize` | `--experiment-run-id` | ตรวจหลักฐานและสร้างสรุป ไม่เริ่มฝึกโมเดล |

Resume รองรับทั้ง `failed` และ `running` ที่ process เดิมหยุดแล้ว โดยต้องได้สิทธิ์ครอบครอง Run ผ่าน OS lock ก่อน ห้ามเริ่มสอง process บน Run เดียวกัน ห้ามลบ lock หรือ complete marker เพื่อข้ามการตรวจ และห้ามแก้ Config/ผลลัพธ์เก่าเพื่อให้ Resume ผ่าน

## Config ที่กำหนดไว้

```text
configs/
├── stage1_baselines.json
├── stage2_wave_a.json
├── stage2_wave_b_features.json
├── stage2_wave_b_augmentation.json
├── stage2_wave_b_loss.json
└── stage2_wave_c.json
```

Wave A มี Candidate 9 ตัวและ seed 42; Wave C บังคับ seeds `42`, `123`, `2026` อย่างเคร่งครัด

ไฟล์ `configs/baselines.json` และ `configs/ablations.json` เป็น catalog นิยามจากโครงสร้างเดิม ไม่ใช่ Config สำหรับ `--config` ของ Shared Engine ให้ใช้หกไฟล์ในรายการข้างต้น โดย Wave B/C ต้องอ้าง parent ที่ตรวจแล้วก่อน Prepare

## โครงสร้าง Experiment Run

```text
runs/<experiment_run_id>/
├── protocol.json
├── inputs.json
├── source_config.json
├── reference_run.json
├── parent_provenance.json
├── record_snapshot.json
├── resolved_config.json
├── shared_fold_assignments.csv
├── candidate_matrix.json
├── integrity.json
├── preparation_integrity.json
├── environment.json
├── state.json
├── .execution.lock
├── candidates/
│   └── <candidate_id>/
│       └── seed_<seed>/
│           ├── fold_1/ ... fold_5/
│           │   ├── attempt_1/
│           │   │   ├── model.keras หรือ model.joblib
│           │   │   ├── fold_manifest.json
│           │   │   └── fold_result.json
│           │   └── complete.json หรือ attempt_N/failure.json
│           ├── oof_predictions.csv
│           ├── oof_metrics.json
│           ├── seed_summary.json
│           └── verification.json
├── leaderboard.json
├── leaderboard.csv
├── leaderboard.md
├── selection.json
├── promotion_recommendation.json
├── promotion_recommendation.md
├── comparison.md
└── verification.json
```

- `prepared`: audit และ immutable snapshots พร้อม แต่ยังไม่เริ่ม Candidate training
- `running`: เริ่มอย่างน้อยหนึ่ง job แล้ว
- `failed`: มี protocol failure หรือ job failure ที่หยุด Run; failure evidence เดิมไม่ถูกลบเมื่อ resume
- `complete`: jobs, OOF aggregation, leaderboard และ reports ผ่าน verification ครบแล้ว
- `no_promotion_recommended`: Run อาจสมบูรณ์ แต่ยังไม่ผ่านเกณฑ์เสนอ Training 3

เมื่อมีงานล้มเหลว สรุปชั่วคราวอยู่ใน `summary_attempt_N/` พร้อม Leaderboard ของ Candidate ที่ตรวจครบและรายการ exclusions โดย Run ยังไม่ถือว่า complete การสร้างสรุปครั้งใหม่ไม่เขียนทับครั้งก่อน ระหว่างรวม OOF/เผยแพร่รายงานอาจมี `aggregation_attempt_N/` และ `report_attempt_N/` เพื่อเก็บหลักฐานการทำงานที่หยุดกลางทาง

`complete.json` ระบุ attempt ที่ใช้จริง ไม่เลือกไฟล์จากเวลาที่แก้ล่าสุด ผล OOF, Config, โมเดล และรายงานต้องผ่านการตรวจ hash ก่อนใช้ต่อ Run ที่ complete แล้วใช้ตรวจ/อ่านได้ แต่เขียนทับหรือเริ่มเทรนซ้ำใน Run เดิมไม่ได้

## กติกาคะแนน

- ตรวจ probability เป็น `float64`; ยอม normalize เฉพาะความคลาดเคลื่อนผลรวมไม่เกิน `1e-5`
- original development record ทุกตัวต้องมี OOF prediction เพียงครั้งเดียวและครบ Fold 1–5
- Screening เรียง Macro F1 แล้วใช้ minimum class recall, parameter count และ Candidate ID เป็น tie-break
- `parameter_count` ของ Neural architecture เดิมต้องเท่ากันทุก Fold ส่วน Classical ใช้จำนวนค่าที่ fit จริง ซึ่ง SVM เปลี่ยนได้ตาม support vectors จึงเก็บค่าราย Fold/ต่ำสุด/สูงสุดและใช้ค่าเฉลี่ยโดยไม่ปัดเป็นจำนวนเต็มในการตัดสินกรณีคะแนนใกล้กัน ไม่ใช่เหตุให้ปฏิเสธผล SVM
- Wave C ใช้ค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐานจากสาม seeds
- Promotion ต้องเพิ่ม Macro F1 อย่างน้อย `0.01`, Balanced Accuracy ไม่ลด, minimum recall ลดไม่เกิน `0.02` และ verification ต้อง complete

## GPU และ checkpoint

TensorFlow อาจแจ้ง PTX JIT warning สำหรับ RTX 5070 Ti compute capability 12.0a ในการสร้าง kernel ครั้งแรก ซึ่งเป็น warning ไม่ใช่ training failure Mutable Keras checkpoint ถูกเขียนบน native Linux staging ก่อนเผยแพร่เข้า immutable Run folder พร้อมตรวจ SHA-256

## รายงาน

- [Experiment Report Hub](../../Report/experiments/report.md)
- [Architecture and protocol](../../Architecture.md)
- [README หลัก](../../README.md)

ผลที่ยังไม่ได้รันต้องรายงานเป็น `not_run` ห้ามใส่ค่า Accuracy ประมาณหรือ metric จำลอง

สถานะการตรวจรับโค้ดอยู่ใน [Inline Execution Status](../../docs/superpowers/plans/2026-09-03-inline-execution-status.md) ไม่ใช่ผลการเทรน และไม่ใช้ผล CPU แทน GPU/checkpoint smoke บน WSL
