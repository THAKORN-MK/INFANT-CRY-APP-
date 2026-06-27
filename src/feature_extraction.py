import librosa
import numpy as np

def extract_melspectrogram(file_path, sr=22050, n_mels=128, max_len=128):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=2048, hop_length=512,
        fmin=50, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0,0),(0, max_len - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :max_len]
    
    return mel_db[..., np.newaxis].astype(np.float32)