# Shared Experiment Engine — สถานะการเก็บงาน Inline Execution

อัปเดต: 2026-09-03

อ้างอิง [สเปกที่อนุมัติ](../specs/2026-08-22-shared-experiment-engine-design.md) และ [แผนเดิม](2026-08-22-shared-experiment-engine.md)

เอกสารนี้เป็นบันทึกสถานะ implementation ไม่ใช่ผลทดลองโมเดล และไม่มีการประมาณ Accuracy

## ขอบเขต

- ต่อจากโค้ดเดิมใน workspace ไม่เริ่มโครงการใหม่
- แก้การรวมผล SVM, provenance/config checks, resume, การส่งต่อ Wave และเอกสาร
- ทดสอบด้วยข้อมูลจำลองขนาดเล็กเท่านั้น ไม่ Prepare/Train Dataset จริง
- ไม่แก้ trainer หลัก, Dataset, deployment หรือ completed pipeline run
- ไม่ commit หรือ push โดยอัตโนมัติ

## สถานะขณะเริ่มเก็บงาน

| รายการ | สถานะ |
|---|---|
| Core Engine และ regression tests | กำลังเก็บจุดตกหล่น |
| README / Architecture / คำสั่ง WSL | กำลังปรับให้ตรง implementation |
| Independent code review | ยังไม่เสร็จ |
| Windows Python 3.10 CPU tests | Baseline 66 tests, 4 errors จาก fsync บน read-only descriptor |
| WSL GPU / checkpoint smoke | ยังตรวจซ้ำไม่ได้ |
| Experiment จริง | ไม่ได้เริ่ม; runs มีเพียง .gitkeep |

ผลตรวจระหว่างเก็บงาน: regression ของส่วนเดิมบน Windows Python 3.10 รัน 92 tests ไม่พบ failure/error และข้าม 1 test; audit ของ Config `stage*.json` ทั้งหกไฟล์ exit 0 โดยไม่เริ่ม training ส่วน `baselines.json` และ `ablations.json` เป็น legacy catalogs ไม่ใช่ runnable Config

## ข้อจำกัดของการตรวจ WSL ในครั้งนี้

การเรียก Ubuntu-22.04 แจ้ง `HCS_E_SERVICE_NOT_AVAILABLE` จากการตรวจแบบอ่านอย่างเดียว:

- WSL 2.6.3 และ Ubuntu-22.04 (WSL 2) ยังปรากฏในรายการ
- `WslService` ทำงาน แต่ไม่พบ `vmcompute` ในการค้น service
- CPU รายงาน `VirtualizationFirmwareEnabled=True`
- Windows รายงาน `HypervisorPresent=False`
- การอ่าน Optional Features และ boot configuration ต้องใช้สิทธิ์ Administrator ซึ่ง session นี้ไม่มี

จึงยังสรุปไม่ได้ว่า Windows feature ใดถูกปิดหรือ boot policy ถูกเปลี่ยน ไม่ติดตั้งใหม่ ไม่แก้ BIOS/boot/Windows features และไม่ reboot โดยอัตโนมัติ ผล CPU ไม่ใช้แทนหลักฐาน GPU/WSL

คำสั่งอ่านสถานะและจุดตรวจ Virtual Machine Platform / boot policy อ้างอิง [Microsoft WSL troubleshooting](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting) ขั้นตอนอ่านข้อมูลที่ต้องใช้สิทธิ์ Administrator อยู่ใน `file.txt` หัวข้อ 17

## สิ่งที่ต้องมีเพื่อปิดงาน

- ผล regression/integration tests ล่าสุด พร้อมจำนวนที่ผ่าน/ข้าม/ล้มเหลว
- ผล audit ของ config จริงโดยไม่สร้าง Run
- ผล code review และประเด็นที่ยังเปิดอยู่
- เปรียบเทียบ SHA-256 ของ trainer หลัก, README_OLD และ completed run 201 ไฟล์กับก่อนเริ่ม
- ผล GPU/checkpoint smoke ใน WSL หรือระบุชัดว่ายังถูกบล็อก
