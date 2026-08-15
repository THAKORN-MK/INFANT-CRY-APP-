import tkinter as tk
from tkinter import filedialog

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def select_audio_file() -> str:
    """เปิดหน้าต่างให้ผู้ใช้เลือกไฟล์เสียง แล้วคืนค่า path"""
    root = tk.Tk()
    root.withdraw()  # ไม่ต้องแสดงหน้าต่างหลักของ tkinter
    file_path = filedialog.askopenfilename(
        title="เลือกไฟล์เสียง",
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path


def plot_mfcc(file_path: str, n_mfcc: int = 13, sr: int | None = None):
    """โหลดไฟล์เสียงแล้ว plot MFCC เป็น heatmap"""
    # โหลดเสียง (sr=None คือใช้ sample rate เดิมของไฟล์)
    y, sample_rate = librosa.load(file_path, sr=sr)

    # คำนวณ MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)

    # วาดกราฟ
    plt.figure(figsize=(12, 4))
    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sample_rate,
        cmap="jet",
    )
    plt.ylabel("MFCC Coefficients")
    plt.xlabel("Time (s)")
    plt.yticks(np.arange(0, n_mfcc, 2))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = select_audio_file()
    if not path:
        print("ไม่ได้เลือกไฟล์ ยกเลิกการทำงาน")
    else:
        print(f"กำลังประมวลผลไฟล์: {path}")
        plot_mfcc(path)