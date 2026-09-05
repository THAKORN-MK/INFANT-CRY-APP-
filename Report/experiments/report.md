# Experiment Reports

หน้านี้เป็นศูนย์รวมรายงานการเปรียบเทียบโมเดลของ Shared Experiment Engine โดยใช้ผลจาก corrected grouped out-of-fold (OOF) validation เท่านั้น

## ประเภทรัน

- **Pipeline run** คือการเทรน Stage 1 และ Stage 2 ที่จับคู่ด้วย Run ID เดียวกัน และเป็นแหล่งอ้างอิงของ fold assignments ที่ถูกตรึงไว้
- **Experiment run** คือการเปรียบเทียบ baseline, ablation หรือ candidate หลายแบบภายใต้ fold assignments เดียวกัน โดยมี ID ขึ้นต้นด้วย Pipeline Run ID

## หลักการรายงาน

- ใช้ grouped OOF สำหรับจัดอันดับ เลือก candidate และตรวจ Promotion Gate
- Final Test ไม่ถูกเปิดใช้ในการทดลอง การจัดอันดับ หรือการเลือกโมเดล
- Candidate ที่ล้มเหลวหรือ verification ไม่สมบูรณ์จะแสดงในรายการ exclusions แต่ไม่เข้าสู่ leaderboard
- จะเพิ่มลิงก์รายงานของรันจริงหลังรันนั้นเสร็จสมบูรณ์และได้รับอนุมัติให้อัปเดตเอกสารแล้วเท่านั้น

ขณะนี้ยังไม่มี Shared Experiment Run ที่เสร็จสมบูรณ์ จึงยังไม่มีลิงก์ผลการทดลองจริงในหน้านี้
