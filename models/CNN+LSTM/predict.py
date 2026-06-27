import numpy as np
import json
import tensorflow as tf
import librosa
import glob

# โหลดโมเดลและข้อมูลที่จำเป็น
model = tf.keras.models.load_model('best_model.keras')
labels = json.load(open('labels.json'))
norm  = np.load('norm_stats.npy')
mean, std = norm[0], norm[1]

LABEL_TH = {
    'belly_pain': 'ปวดท้อง 🤕',
    'burping':    'เรอ 😮‍💨',
    'discomfort': 'ไม่สบาย 😣',
    'hungry':     'หิว 🍼',
    'tired':      'เหนื่อย/ง่วง 😴'
}

def predict(wav_path):
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)
    y     = librosa.util.normalize(y)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128,
        n_fft=2048, hop_length=512, fmin=50, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0,0),(0, 128 - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :128]

    feat = mel_db[..., np.newaxis].astype(np.float32)
    feat = (feat - mean) / std
    feat = feat[np.newaxis, ...]

    probs   = model.predict(feat, verbose=0)[0]
    top_idx = np.argmax(probs)
    top_lbl = labels[top_idx]

    print(f"\n{'='*45}")
    print(f"  ไฟล์  : {wav_path}")
    print(f"  ผลลัพธ์: {LABEL_TH[top_lbl]}  ({probs[top_idx]*100:.1f}%)")
    print(f"{'='*45}")
    for lbl, p in zip(labels, probs):
        bar = '█' * int(p * 30)
        print(f"  {LABEL_TH[lbl]:22s} {bar:<30s} {p*100:5.1f}%")
    print()
    return top_lbl, float(probs[top_idx])

# หยิบไฟล์อัตโนมัติ ไม่ต้องพิมพ์ชื่อเอง
for emotion in ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']:
    files = glob.glob(f'data_set/{emotion}/*.wav')
    if files:
        print(f"\n--- ทดสอบ class: {emotion} ---")
        predict(files[0])
    else:
        print(f"  ไม่พบไฟล์ใน data_set/{emotion}/")