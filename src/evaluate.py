import numpy as np
import json
import glob
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ---- โหลด Font ภาษาไทย ----
# ดาวน์โหลด Sarabun font ก่อน
import urllib.request
import os

font_path = 'Sarabun-Regular.ttf'
if not os.path.exists(font_path):
    print("ดาวน์โหลด font ภาษาไทย...")
    urllib.request.urlretrieve(
        'https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf',
        font_path
    )

prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
fm.fontManager.addfont(font_path)

print("โหลดโมเดล...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
model  = tf.keras.models.load_model('best_model_yamnet.keras')
labels = json.load(open('labels_yamnet.json'))
norm   = np.load('norm_stats_yamnet.npy')
mean, std = norm[0], norm[1]

LABEL_TH = {
    'belly_pain': 'ปวดท้อง',
    'burping':    'เรอ',
    'discomfort': 'ไม่สบาย',
    'hungry':     'หิว',
    'tired':      'เหนื่อย'
}

# โหลดข้อมูลทดสอบ
X_test, y_test = [], []
for label in labels:
    files = glob.glob(f'data_set/{label}/*.wav')
    test_files = files[:max(1, len(files)//5)]
    for f in test_files:
        try:
            audio, sr = librosa.load(f, sr=16000, mono=True)
            audio, _  = librosa.effects.trim(audio, top_db=20)
            audio     = librosa.util.normalize(audio).astype(np.float32)
            _, emb, _ = yamnet_model(audio)
            emb = np.mean(emb.numpy(), axis=0)
            emb = (emb - mean) / std
            X_test.append(emb)
            y_test.append(label)
        except:
            pass

X_test = np.array(X_test)
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = [labels[np.argmax(p)] for p in y_pred_probs]

label_names_th = [LABEL_TH[l] for l in labels]

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_names_th))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True, fmt='d',
    xticklabels=label_names_th,
    yticklabels=label_names_th,
    cmap='Blues',
    linewidths=0.5
)
plt.title('Confusion Matrix — Baby Cry Emotion Detection', fontproperties=prop, fontsize=14, pad=15)
plt.ylabel('ค่าจริง (Actual)', fontproperties=prop, fontsize=12)
plt.xlabel('ค่าที่ทำนาย (Predicted)', fontproperties=prop, fontsize=12)
plt.xticks(fontproperties=prop, fontsize=11)
plt.yticks(fontproperties=prop, fontsize=11, rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("\n✅ บันทึก confusion_matrix.png แล้ว!")