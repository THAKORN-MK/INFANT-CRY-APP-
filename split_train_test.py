"""
split_train_test.py

แบ่งไฟล์ .wav ใน data_set_dbl แบบสุ่ม (stratified ต่อคลาส) ออกเป็น train/test
- แบ่ง "ก่อน" augment เสมอ (สคริปต์นี้ไม่ augment อะไรเลย แค่คัดลอกไฟล์ดิบ)
- ไม่แตะ/ไม่ลบไฟล์ต้นฉบับใน DATA_DIR — คัดลอกไปโฟลเดอร์ใหม่เท่านั้น
- ใช้ seed ตายตัว (random_state=42) เพื่อให้แบ่งซ้ำได้ผลเดิมทุกครั้ง (reproducible)
"""

import os
import glob
import shutil
import json
from sklearn.model_selection import train_test_split

# ========== CONFIG ==========
BASE_DIR   = 'D:/INFANT CRY'
DATA_DIR   = f'{BASE_DIR}/data_set_dbl'          # โฟลเดอร์ต้นฉบับ (ไม่ถูกแก้ไข)
OUT_DIR    = f'{BASE_DIR}/data_set_dbl_split'    # โฟลเดอร์ผลลัพธ์ใหม่
TRAIN_DIR  = f'{OUT_DIR}/train'
TEST_DIR   = f'{OUT_DIR}/test'
EMOTIONS   = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired','not_baby']
TEST_SIZE  = 0.2          # 80/20
RANDOM_STATE = 42
# =============================


def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    summary = {}
    print(f"📂 อ่านไฟล์จาก: {DATA_DIR}\n")

    for label in EMOTIONS:
        files = sorted(glob.glob(f'{DATA_DIR}/{label}/*.wav'))
        if len(files) == 0:
            print(f"  ⚠️  {label}: ไม่พบไฟล์ .wav ข้ามคลาสนี้")
            continue

        # stratify ทำที่ระดับไฟล์อยู่แล้วเพราะแบ่งทีละคลาส (เท่ากับ stratify โดยธรรมชาติ)
        train_files, test_files = train_test_split(
            files,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        # เตรียมโฟลเดอร์ปลายทางต่อคลาส
        train_class_dir = f'{TRAIN_DIR}/{label}'
        test_class_dir  = f'{TEST_DIR}/{label}'
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for f in train_files:
            shutil.copy2(f, os.path.join(train_class_dir, os.path.basename(f)))
        for f in test_files:
            shutil.copy2(f, os.path.join(test_class_dir, os.path.basename(f)))

        summary[label] = {
            'total': len(files),
            'train': len(train_files),
            'test': len(test_files),
        }

        print(f"  {label:12s}: total={len(files):4d}  "
              f"-> train={len(train_files):4d}  test={len(test_files):4d}")

    # บันทึกสรุปเป็น JSON ไว้อ้างอิง/ใส่ในรายงาน
    summary_path = f'{OUT_DIR}/split_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'per_class': summary,
            'total_train': sum(v['train'] for v in summary.values()),
            'total_test': sum(v['test'] for v in summary.values()),
        }, f, indent=2, ensure_ascii=False)

    total_train = sum(v['train'] for v in summary.values())
    total_test  = sum(v['test'] for v in summary.values())

    print(f"\n{'='*50}")
    print(f"✅ แบ่งเสร็จ")
    print(f"  Train รวม : {total_train} ไฟล์  -> {TRAIN_DIR}")
    print(f"  Test  รวม : {total_test} ไฟล์  -> {TEST_DIR}")
    print(f"  Summary   : {summary_path}")
    print(f"{'='*50}")
    print(f"\n⚠️  ขั้นถัดไป: ใช้ {TRAIN_DIR} เป็น input ให้ augment_audio() ในสคริปต์เทรน")
    print(f"   ส่วน {TEST_DIR} ห้าม augment เด็ดขาด — ใช้ extract_features() ตรงๆ")
    print(f"   แล้ว evaluate บนชุดนี้ครั้งเดียวหลังเทรนเสร็จ (independent test)")


if __name__ == '__main__':
    main()