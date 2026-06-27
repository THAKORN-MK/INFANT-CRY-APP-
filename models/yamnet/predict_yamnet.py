import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import glob

print("โหลดโมเดล...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
model  = tf.keras.models.load_model('best_model_yamnet.keras')
labels = json.load(open('labels_yamnet.json'))
norm   = np.load('norm_stats_yamnet.npy')
mean, std = norm[0], norm[1]

LABEL_TH = {
    'belly_pain': 'ปวดท้อง 🤕',
    'burping':    'เรอ 😮‍💨',
    'discomfort': 'ไม่สบาย 😣',
    'hungry':     'หิว 🍼',
    'tired':      'เหนื่อย/ง่วง 😴'
}

def predict(wav_path):
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    audio, _  = librosa.effects.trim(audio, top_db=20)
    audio     = librosa.util.normalize(audio).astype(np.float32)

    _, embeddings, _ = yamnet_model(audio)
    emb = np.mean(embeddings.numpy(), axis=0)
    emb = (emb - mean) / std
    emb = emb[np.newaxis, ...]

    probs   = model.predict(emb, verbose=0)[0]
    top_idx = np.argmax(probs)
    top_lbl = labels[top_idx]

    print(f"\n{'='*50}")
    print(f"  ไฟล์    : {wav_path}")
    print(f"  ผลลัพธ์ : {LABEL_TH[top_lbl]}  ({probs[top_idx]*100:.1f}%)")
    print(f"{'='*50}")
    for lbl, p in zip(labels, probs):
        bar = '█' * int(p * 30)
        print(f"  {LABEL_TH[lbl]:22s} {bar:<30s} {p*100:5.1f}%")
    print()
    return top_lbl, float(probs[top_idx])

# ทดสอบทุก class อัตโนมัติ
for emotion in ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']:
    files = glob.glob(f'data_set/{emotion}/*.wav')
    if files:
        print(f"\n--- ทดสอบ class: {emotion} ---")
        predict(files[0])
    else:
        print(f"  ไม่พบไฟล์ใน data_set/{emotion}/")