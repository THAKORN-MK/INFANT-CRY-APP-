# CryInsight Experiment Report Hub

หน้านี้เป็นศูนย์รวมรายงานการฝึกและประเมินโมเดล CryInsight ทั้ง Stage 1 และ Stage 2 ผลที่รายงานต้องมาจาก immutable run ที่ `verification.json` มีสถานะ `complete` เท่านั้น

> ผลทั้งหมดเป็นการประเมินภายในโครงการ ไม่ใช่ external validation และไม่ใช่ผลวินิจฉัยทางการแพทย์

## Pipeline Run 1 — ผลหลักปัจจุบัน

Pipeline Run `20260821T164332Z_490383ff` เป็นผลหลักหลังรื้อและจัดระเบียบระบบใหม่ โดย Stage 1 และ Stage 2 เทรนบน WSL2 Ubuntu 22.04 ด้วย `GPU:0`, `--require-gpu` และ mixed precision จริง

| รายการ | Stage 1 — Binary Baby Gate | Stage 2 — Five-class Infant State |
|---|---:|---:|
| สถานะ | `complete` | `complete` |
| OOF support | 2,432 | 1,045 |
| OOF accuracy | 98.93% | 91.10% |
| Locked internal Test support | 694 | 303 |
| Locked internal Test accuracy | 99.42% | 92.41% |
| Locked internal Test Macro F1 | 99.41% | 91.02% |
| Test evaluation count | 1 | 1 |

อ่านรายละเอียด metric, confidence interval, ผลรายคลาส, GPU provenance และข้อจำกัดได้ที่ [Pipeline Run 1 Report](./runs/report_01_20260821.md)

## สถานะ Baseline และ Ablation

Shared Experiment Engine ใช้ grouped OOF เพื่อจัดอันดับ Candidate และไม่เปิด locked Test ระหว่าง Wave A, B หรือ C

| รายการ | สถานะล่าสุด | รายงาน |
|---|---|---|
| Stage 1 baselines | `prepared` — 15 candidate-fold jobs, ยังไม่เริ่ม training | [Baseline Experiment Status](./experiments/baseline_report.md) |
| Stage 2 baselines / architecture screening | `not_run` | [Baseline Experiment Status](./experiments/baseline_report.md) |
| Feature, augmentation, loss ablations และ repeated seeds | `not_run` | [Ablation Experiment Status](./experiments/ablation_report.md) |

`prepared` หมายถึงระบบตรึง config, record snapshot และ fold-assignment hashes แล้ว ไม่ใช่ผลเทรนและไม่มี metric ให้ตีความ จนกว่าจะเกิด `verification.json` สถานะ `complete` ของ Experiment Run

## เอกสารวิธีดำเนินงาน

- [บทที่ 2 เอกสารและงานวิจัยที่เกี่ยวข้อง](./chapter_2_literature_review.md)
- [บทที่ 3 วิธีการดำเนินงาน](./chapter_3_methodology.md)
- [README หลัก](../README.md)
- [Model Architecture and Training Protocol](../Architecture.md)

## ประวัติรายงาน

| รายการ | วันที่ | ขอบเขต | สถานะ | รายงาน |
|---|---|---|---|---|
| Pipeline Run 1 | 21 สิงหาคม 2026 (UTC Run ID) | Stage 1 และ Stage 2, GPU training | `complete` | [เปิดรายงาน Run 1](./runs/report_01_20260821.md) |
| Legacy archive | 16 สิงหาคม 2026 | ระบบก่อนรื้อและจัดระเบียบใหม่ | historical | [เปิดรายงาน legacy](./runs/report_01_20260816.md) |

## หลักการจัดเก็บรายงาน

- `report.md` เป็นหน้า Hub และชี้ไปยังผลหลักล่าสุด
- Pipeline Run ID เป็นหลักฐานต้นทางของ Stage 1/Stage 2; Experiment Run ID เป็นหลักฐานย่อยของแต่ละ Wave
- รายงานเก่าเก็บไว้เป็น historical record และไม่ใช้แทนผลของ Pipeline Run 1
- ผล Test ใช้รายงาน Final-refit เพียงครั้งเดียว ไม่ใช้เลือก candidate, hyperparameter หรือ deployment model
