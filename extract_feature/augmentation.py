import numpy as np
import librosa

def noise(data):
    """Thêm nhiễu Gaussian vào tín hiệu âm thanh."""
    if data is None or len(data) == 0:
        return data
    max_val = np.amax(data)
    noise_amp = 0 if max_val == 0 else 0.035 * np.random.uniform() * max_val
    return np.clip(data + noise_amp * np.random.normal(size=data.shape[0]), -1, 1)

def stretch(data, rate=0.8):
    """Thay đổi tốc độ phát của tín hiệu âm thanh."""
    if data is None or len(data) == 0:
        return data
    try:
        return librosa.effects.time_stretch(y=data, rate=rate)
    except Exception as e:
        print(f"Error during stretch: {e}")
        return data

def shift(data):
    """Dịch chuyển tín hiệu âm thanh."""
    if data is None or len(data) == 0:
        return data
    max_shift = int(0.2 * len(data))
    shift_range = np.random.randint(-max_shift, max_shift)
    return np.roll(data, shift_range)

def pitch(data, sr, n_steps=0.7):
    """Thay đổi cao độ của tín hiệu âm thanh."""
    if data is None or len(data) == 0 or n_steps == 0:
        return data
    try:
        return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)
    except Exception as e:
        print(f"Error during pitch: {e}")
        return data