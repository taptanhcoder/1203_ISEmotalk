import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from .extract_librosa import extract_features
from .augmentation import noise, pitch, shift, stretch
from typing import Union, List, Tuple
import utils
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_path(path: str, train: bool = True) -> List[Tuple[str, int]]:
    """
    Tải dữ liệu và trả về danh sách chứa đường dẫn file và nhãn cảm xúc.

    Args:
        path (str): Đường dẫn tới thư mục chứa dataset hoặc một file đơn.
        train (bool): Nếu True, lấy nhãn từ tên file. Nếu False, gán nhãn là -1.

    Returns:
        List[Tuple[str, int]]: Danh sách chứa (đường dẫn file, nhãn cảm xúc).
    """
    dataset_path = Path(path)


    if dataset_path.is_file():
        print(f" Đang xử lý file đơn lẻ: {dataset_path}")
        return [(str(dataset_path), -1)]  


    if not dataset_path.exists():
        raise FileNotFoundError(f"[Error] Dataset folder '{dataset_path}' không tồn tại!")

    result_list = []  
    for file in dataset_path.rglob("*.wav"):  
        try:
            if train:
                part = file.stem.split("-")
                emotion_label = int(part[2]) - 1
            else:
                emotion_label = -1

            result_list.append((str(file), emotion_label))

        except (IndexError, ValueError):
            print(f"[Warning] Không thể đọc thông tin từ file: {file}")
            continue

    if not result_list:
        raise ValueError("[Error] Không tìm thấy file âm thanh nào trong dataset!")    

    return result_list


#  -1:'predict',   0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fear', 6: 'Disgust', 7: 'Surprise'


def get_max_min(files: list) -> Tuple[float]:
    if len(files) == 1:
        return 100, 0 

    min_, max_ = 100, 0
    for file in files:
        sound_file, samplerate = librosa.load(file, sr=None)
        t = sound_file.shape[0] / samplerate
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_



def prepara_data(config, train: bool) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    feature_path = os.path.join(config.features.feature_folder, "train.p" if train else "predict.p")

    
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"[Error] File chứa đặc trưng '{feature_path}' không tồn tại!")

    try:
        
        data = joblib.load(feature_path)
        if isinstance(data, tuple) and len(data) == 2:
            features, labels = data
        else:
            raise ValueError("[Error] Dữ liệu trong feature file không đúng định dạng.")
        
        
        max_len = max(f.shape[0] for f in features)  
        features = np.array([np.pad(f, (0, max_len - f.shape[0])) if f.shape[0] < max_len else f for f in features])

        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        
        print("Feature shape:", features.shape)
        print("Label shape:", labels.shape)

    except Exception as e:
        raise ValueError(f"[Error] Lỗi khi tải feature file: {e}")


    if len(features) == 0 or len(labels) == 0:
        raise ValueError("[Error] Dữ liệu đặc trưng hoặc nhãn bị rỗng!")

    if len(features) != len(labels):
        raise ValueError("[Error] Số lượng đặc trưng và nhãn không khớp!")

    
    scaler_path = os.path.join(config.checkpoint.checkpoint_path, 'SCALER_LIBROSA.m')

    if train:
        
        scaler = StandardScaler().fit(features)
        os.makedirs(config.checkpoint.checkpoint_path, exist_ok=True)  
        joblib.dump(scaler, scaler_path)
        features = scaler.transform(features)

        
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)


        return x_train, x_test, y_train, y_test
    else:
        
        scaler = joblib.load(scaler_path)
        features = scaler.transform(features)

        return features


def get_features(paths_list: List[Tuple[str, int]], config=None, train: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Tải file âm thanh từ danh sách đường dẫn và nhãn cảm xúc, trích xuất đặc trưng và lưu lại.

    Args:
        paths_list (List[Tuple[str, int]]): Danh sách chứa (đường dẫn file, nhãn cảm xúc).
        config (dict, optional): Cấu hình thông số âm thanh.
        train (bool): Nếu True, thực hiện augmentation và lưu đặc trưng.

    Returns:
        Union[Tuple[np.ndarray], np.ndarray]: Trả về đặc trưng đã được lưu hoặc gọi `prepara_data()`
    """

    if len(paths_list) == 1:
        print(f" Đang dự đoán cho 1 file: {paths_list[0][0]}")

    max_duration, _ = get_max_min([path for path, _ in paths_list])

    all_features = []  
    all_labels = []    

    for path, label in paths_list:
        sr = config.features.sr if config else 22050
        duration = max_duration 
        offset = config.features.offset if config else 0.6

        try:
            if not os.path.exists(path):
                print(f"[Warning] File không tồn tại: {path}")
                continue

            data, sample_rate = librosa.load(path, sr=sr, duration=duration, offset=offset)
            if data is None or len(data) == 0:
                print(f"[Warning] File rỗng: {path}")
                continue

            if train:
                augmented_features = []
                augmented_data = [
                    data,  
                    noise(data), 
                    stretch(data), 
                    shift(data), 
                    pitch(data, sample_rate), 
                    noise(pitch(data, sample_rate))  
                ]

                for aug_data in augmented_data:
                    extracted_feature = extract_features(aug_data, sample_rate)
                    if extracted_feature is not None and extracted_feature.shape[0] > 0:
                        augmented_features.append(extracted_feature)

                if len(augmented_features) > 0:
                    features = np.vstack(augmented_features)
                    all_labels.extend([label] * features.shape[0])  
                    all_features.append(features)
                
                max_length = max(f.shape[1] for f in all_features)
                all_features = [np.pad(f, ((0, 0), (0, max_length - f.shape[1]))) if f.shape[1] < max_length else f for f in all_features]

            else: 
                features = extract_features(data, sample_rate)
                if features is not None and features.shape[0] > 0:
                    all_labels.append(label)
                    all_features.append(features)

        except Exception as e:
            print(f"[Error] Lỗi khi xử lý file {path}: {e}")
            continue

    if len(all_features) == 0:
        print("[Error] Không có dữ liệu đặc trưng hợp lệ.")
        return None, None


    utils.mkdirs(config.features.feature_folder)
    feature_path = os.path.join(config.features.feature_folder, "train.p" if train else "predict.p")

    try:
        with open(feature_path, "wb") as f:
            pickle.dump((np.vstack(all_features), np.array(all_labels)), f)
    except Exception as e:
        print(f"[Error] Lỗi khi lưu feature file: {e}")
        return None, None

    return prepara_data(config, train=train)


        