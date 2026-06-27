import numpy as np
import json
import tensorflow as tf
import librosa
import glob

# ══════════════════════════════════════════
#  Custom objects สำหรับโหลดโมเดล
# ══════════════════════════════════════════
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True, name='attn_W')
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True, name='attn_b')
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer='glorot_uniform', trainable=True, name='attn_u')
    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        score = tf.tensordot(score, self.u, axes=1)
        alpha = tf.nn.softmax(score, axis=1)
        alpha = tf.expand_dims(alpha, -1)
        return tf.reduce_sum(x * alpha, axis=1)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn

# ══════════════════════════════════════════
#  โหลดโมเดล
# ══════════════════════════════════════════
model  = tf.keras.models.load_model(
    'best_model_CNN_MFCC2.keras',
    custom_objects={'AttentionLayer': AttentionLayer, 'loss_fn': focal_loss()}
)
labels = json.load(open('labels_CNN_MFCC2.json'))
norm   = np.load('norm_stats_CNN_MFCC2.npy')
mean, std = norm[0], norm[1]

LABEL_TH = {
    'belly_pain': 'ปวดท้อง 🤕',
    'burping':    'เรอ 😮‍💨',
    'discomfort': 'ไม่สบาย 😣',
    'hungry':     'หิว 🍼',
    'tired':      'เหนื่อย/ง่วง 😴'
}

# ══════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════
def predict(wav_path):
    audio, sr = librosa.load(wav_path, sr=22050, mono=True)
    audio, _  = librosa.effects.trim(audio, top_db=20)
    audio     = librosa.util.normalize(audio).astype(np.float32)

    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40,
                                   n_fft=2048, hop_length=512)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])

    if combined.shape[1] < 128:
        combined = np.pad(combined, ((0,0),(0, 128 - combined.shape[1])))
    else:
        combined = combined[:, :128]

    feat = combined[..., np.newaxis].astype(np.float32)
    feat = (feat - mean) / std
    feat = feat[np.newaxis, ...]

    probs   = model.predict(feat, verbose=0)[0]
    top_idx = np.argmax(probs)
    top_lbl = labels[top_idx]

    print(f"\n{'='*45}")
    print(f"  ไฟล์   : {wav_path}")
    print(f"  ผลลัพธ์: {LABEL_TH[top_lbl]}  ({probs[top_idx]*100:.1f}%)")
    print(f"{'='*45}")
    for lbl, p in zip(labels, probs):
        bar = '█' * int(p * 30)
        print(f"  {LABEL_TH[lbl]:22s} {bar:<30s} {p*100:5.1f}%")
    print()
    return top_lbl, float(probs[top_idx])

# ══════════════════════════════════════════
#  ทดสอบทุก class
# ══════════════════════════════════════════
DATA_DIR = 'D:/INFANT CRY/data_set'

for emotion in ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']:
    files = glob.glob(f'{DATA_DIR}/{emotion}/*.wav')
    if files:
        print(f"\n--- ทดสอบ class: {emotion} ---")
        predict(files[0])
    else:
        print(f"  ไม่พบไฟล์ใน {emotion}/")
