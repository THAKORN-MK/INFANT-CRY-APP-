# Ablation Experiment Status

วันที่ปรับปรุง: 21 สิงหาคม 2026

สถานะรวม: `not_run`

โค้ดและ protocol ลงทะเบียนการทดลองต่อไปนี้แล้ว แต่ยังไม่ได้ Full Train จึงยังไม่มีข้อสรุปว่าส่วนประกอบใดเพิ่ม accuracy

| Ablation | Candidate ที่ต้องเปรียบเทียบ | สถานะ |
|---|---|---|
| Recurrent/attention | CNN only; CNN+BiLSTM; CNN+BiLSTM+Attention | `not_run` |
| Stage 2 architecture | corrected single-branch; corrected multi-branch | `not_run` |
| Feature blocks | MFCC derivatives; Log-Mel; Chroma removal | `not_run` |
| Normalization | global scalar; per-feature-bin | `not_run` |
| Augmentation | none; waveform only; waveform+Mixup | `not_run` |

เกณฑ์ development เป้าหมายคือ grouped OOF accuracy ≥97%, Macro-F1 ≥95%, Balanced Accuracy ≥95% และ recall ทุกคลาส ≥90% แต่เป็นเป้าหมาย ไม่ใช่ค่ารับประกัน Candidate ต้องเลือกจาก grouped OOF และห้ามปรับจาก held-out Test เพื่อไล่ตัวเลข

[กลับ Report Hub](../report.md)
