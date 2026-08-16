# CryInsight Experiment Report Hub

หน้านี้เป็นศูนย์รวมรายงานการฝึกและประเมินโมเดล CryInsight ทั้ง Stage 1 และ Stage 2 รายงานแต่ละครั้งเก็บแยกเป็นไฟล์เพื่อรักษาประวัติผลการทดลองและป้องกันการเขียนทับผลเดิม

> ผลทั้งหมดในหน้านี้เป็นการประเมินภายในโครงการ ไม่ใช่ external validation และไม่ใช่ผลวินิจฉัยทางการแพทย์

## ผลล่าสุด

| รายการ | Stage 1 — Binary Baby Gate | Stage 2 — Five-class Infant State |
|---|---:|---:|
| Run ID | `20260816T092253Z_e1fe86b3` | `20260816T102950Z_a16c5a1d` |
| สถานะ | `complete` | `complete` |
| OOF accuracy | 98.93% | 90.14% |
| Final Test accuracy | 98.99% | 90.76% |
| Final Test Macro F1 | 98.97% | 88.51% |
| Final Test support | 694 | 303 |

รายละเอียดวิธีทดลอง ผลรายคลาส confusion matrix การตรวจความสมบูรณ์ และปัญหาที่พบอยู่ในรายงานฉบับเต็ม

## ประวัติรายงาน

| ครั้ง | วันที่ | ขอบเขต | สถานะ | รายงาน |
|---:|---|---|---|---|
| 1 | 16 สิงหาคม 2026 | Stage 1 และ Stage 2 | Complete | [เปิดรายงานครั้งที่ 1](./runs/report_01_20260816.md) |

## การนำทาง

- [กลับ README หลัก](../README.md)
- [อ่าน Model Architecture and Training Protocol](../Architecture.md)
- [รายงานการทดลองครั้งที่ 1](./runs/report_01_20260816.md)

## หลักการจัดเก็บรายงาน

- `report.md` เป็นหน้า Hub และสรุปผลล่าสุด
- รายงานแต่ละครั้งใน `runs/` เป็นบันทึกถาวรของ immutable run ที่อ้างอิง
- เมื่อมีการทดลองใหม่ ให้สร้าง `report_02`, `report_03` ตามลำดับ โดยไม่แก้ตัวเลขของรายงานเก่า
- รายงานผลได้เฉพาะ run ที่ `verification.json` มีสถานะ `complete`

