import os, glob
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════
# CONFIG — ต้องตรงกับ train_main_dbl.py
# ══════════════════════════════════════════
DATA_DIR = 'D:/INFANT CRY/data_set_dbl'
SAVE_DIR = 'D:/INFANT CRY/models_dbl/main'
EMOTIONS = ['belly_pain','burping','discomfort','hungry','tired']
SR=22050; N_MFCC=40; N_MELS=64; N_CHROMA=12; MAX_LEN=128; N_FOLDS=5

AUG_TIMES = {
    'belly_pain': 12,
    'burping':     8,
    'discomfort': 12,
    'hungry':     10,
    'tired':       3,
}

def extract_features(audio, sr=SR, max_len=MAX_LEN):
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel    = librosa.power_to_db(librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA, n_fft=2048, hop_length=512)
    combined = np.vstack([mfcc, delta, delta2, mel, chroma])
    if combined.shape[1] < max_len:
        combined = np.pad(combined, ((0,0),(0, max_len - combined.shape[1])))
    else:
        combined = combined[:, :max_len]
    return combined[..., np.newaxis].astype(np.float32)

def augment_audio(y, sr, n=6):
    results = [y]
    results.append(y + 0.004 * np.random.randn(len(y)))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5))
    results.append(librosa.effects.time_stretch(y, rate=1.1))
    results.append(librosa.effects.time_stretch(y, rate=0.9))
    results.append(y * np.random.uniform(0.8, 1.2))
    results.append(y + 0.002 * np.random.randn(len(y)))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0))
    results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.0))
    results.append(librosa.effects.time_stretch(y, rate=1.2))
    results.append(librosa.effects.time_stretch(y, rate=0.8))
    return results[:n]

# โหลด dataset เหมือนตอนเทรน
print("📂 โหลด dataset...")
X, y_labels = [], []
for label in EMOTIONS:
    files = glob.glob(f'{DATA_DIR}/{label}/*.wav')
    aug   = AUG_TIMES[label]
    print(f"  {label:15s}: {len(files)} ไฟล์")
    for f in files:
        try:
            audio, sr = librosa.load(f, sr=SR, mono=True)
            audio, _  = librosa.effects.trim(audio, top_db=20)
            audio     = librosa.util.normalize(audio)
            for aug_audio in augment_audio(audio, sr, n=aug):
                aug_audio = librosa.util.normalize(aug_audio.astype(np.float32))
                X.append(extract_features(aug_audio))
                y_labels.append(label)
        except Exception as e:
            print(f'    Skip: {e}')

X     = np.array(X)
le    = LabelEncoder()
y_int = le.fit_transform(y_labels)

# หา best fold จาก log
import pandas as pd
fold_scores = []
for i in range(1, N_FOLDS+1):
    log = pd.read_csv(f'{SAVE_DIR}/fold_{i}_log.csv')
    fold_scores.append(log['val_accuracy'].max())
    print(f"  Fold {i}: {fold_scores[-1]*100:.2f}%")

best_fold = int(np.argmax(fold_scores)) + 1
print(f"\n⭐ Best fold: {best_fold} ({fold_scores[best_fold-1]*100:.2f}%)")

# คำนวณ norm_stats ของ best fold
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_int)):
    if fold + 1 == best_fold:
        X_train = X[train_idx]
        mean = X_train.mean()
        std  = X_train.std()
        np.save(f'{SAVE_DIR}/norm_stats_main_dbl.npy', [mean, std])
        print(f"\n✅ บันทึก norm_stats ของ fold {best_fold} แล้ว")
        print(f"   mean = {mean:.6f}")
        print(f"   std  = {std:.6f}")
        break

print("\n✅ แก้ปัญหาข้อ 6 เสร็จแล้ว!")
print(f"   best_model = fold_{best_fold}_main.keras")
print(f"   norm_stats = norm_stats_main_dbl.npy (fold {best_fold})")