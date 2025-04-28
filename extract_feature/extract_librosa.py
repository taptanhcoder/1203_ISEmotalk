
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
 

def zcr(data, frame_length=2048, hop_length=512, fixed_size=100):
    """Tính tỷ lệ Zero-Crossing Rate (ZCR)."""
    if data is None or len(data) == 0:
        return np.zeros(fixed_size)
    zcr_values = librosa.feature.zero_crossing_rate(
        y=data, frame_length=frame_length, hop_length=hop_length
    ).squeeze()
    return np.pad(zcr_values, (0, max(0, fixed_size - len(zcr_values))), mode="constant")[:fixed_size]

def rmse(data, frame_length=2048, hop_length=512, fixed_size=100):
    """Tính Root Mean Square Energy (RMSE)."""
    if data is None or len(data) == 0:
        return np.zeros(fixed_size)
    rms_values = librosa.feature.rms(
        y=data, frame_length=frame_length, hop_length=hop_length
    ).squeeze()
    return np.pad(rms_values, (0, max(0, fixed_size - len(rms_values))), mode="constant")[:fixed_size]

def mfcc(data, sr=22050, frame_length=2048, hop_length=512, n_mfcc=13, fixed_size=100):
    """Trích xuất MFCC từ tín hiệu âm thanh."""
    if data is None or len(data) == 0:
        return np.zeros(n_mfcc * fixed_size)
    mfcc_values = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_padded = [np.pad(coeff, (0, max(0, fixed_size - len(coeff))), mode="constant")[:fixed_size] for coeff in mfcc_values]
    return np.array(mfcc_padded).flatten()

def extract_features(data, sr=22050, frame_length=2048, hop_length=512, n_mfcc=13, duration=40, fixed_size=100):
    """Trích xuất đặc trưng tổng hợp từ tín hiệu âm thanh."""
    max_length = int(sr * duration)
    data = np.pad(data, (0, max(0, max_length - len(data))), mode="constant")[:max_length]

    features = np.hstack((zcr(data, frame_length, hop_length, fixed_size),
                          rmse(data, frame_length, hop_length, fixed_size),
                          mfcc(data, sr, frame_length, hop_length, n_mfcc, fixed_size)))

    return features if np.std(features) != 0 else features / np.std(features)