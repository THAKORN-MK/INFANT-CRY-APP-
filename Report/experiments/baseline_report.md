# Baseline Experiment Status

วันที่ปรับปรุง: 4 กันยายน 2026

สถานะรวม: `prepared` เฉพาะ Stage 1 baseline; Stage 2 baseline ยัง `not_run`

Stage 1 baseline ของ Pipeline Run `20260821T164332Z_490383ff` ถูกเตรียมเป็น Experiment Run `20260821T164332Z_490383ff__exp_20260903T170029Z_e0883d24` แล้ว โดยตรึง candidate matrix, input snapshot และ assignment hashes ครบ 15 candidate-fold jobs แต่ยังไม่ได้เริ่ม `--train` และไม่มีการกรอก metric ประมาณ

| Stage | Baseline | Feature | สถานะ |
|---|---|---|---|
| Stage 1 | Majority / most frequent | Labels only | `prepared` |
| Stage 1 | SVM | MFCC summary | `prepared` |
| Stage 1 | Small CNN | Log-Mel | `prepared` |
| Stage 2 | Majority / most frequent | Labels only | `not_run` |
| Stage 2 | SVM | MFCC summary | `not_run` |
| Stage 2 | Small CNN | Log-Mel | `not_run` |

กฎบังคับ:

- ใช้ eligible original cohort, group rules และ fold assignments เดียวกับ proposed model
- fit scaler/normalizer และ augmentation จาก fold Training เท่านั้น
- Validation/OOF เป็น original records เท่านั้น
- จัดอันดับด้วย grouped OOF; locked held-out Test ไม่พร้อมให้ใช้เลือก baseline
- รันอย่างน้อย seeds `42`, `123`, `2026` ก่อนสรุปความแปรปรวน
- รายงานผลเฉพาะ immutable experiment run ที่ verification สมบูรณ์

[กลับ Report Hub](../report.md)
