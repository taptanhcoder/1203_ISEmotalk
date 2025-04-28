import os
import joblib
import numpy as np

FEATURES_PATH = "features/8-category/librosa_ravdess/train.p"
MIN_FEATURE_LENGTH = 5 

def check_feature_size(feature_path=FEATURES_PATH, min_length=MIN_FEATURE_LENGTH, save_fixed=False):
    """
    Kiểm tra kích thước feature trong train.p trước khi training.

    Args:
        feature_path (str): Đường dẫn đến file train.p.
        min_length (int): Độ dài tối thiểu của feature.
        save_fixed (bool): Nếu True, lưu dữ liệu đã sửa vào 'checked_train.p'.

    Returns:
        tuple: Trả về kích thước feature nếu hợp lệ, hoặc None nếu lỗi.
    """
    if not os.path.exists(feature_path):
        print(f"[Error] File không tồn tại: {feature_path}")
        return None

    try:
        # Load dữ liệu từ file train.p
        data = joblib.load(feature_path)

        # Kiểm tra định dạng dữ liệu
        if not isinstance(data, tuple) or len(data) != 2:
            print("[Error] Dữ liệu không đúng định dạng! (Phải là (features, labels))")
            return None

        features, labels = data

        # Kiểm tra kích thước dữ liệu
        if features.shape[0] == 0 or features.shape[1] == 0:
            print("[Error] Dữ liệu đặc trưng bị rỗng!")
            return None

        print(f"[INFO] Kích thước feature: {features.shape}")
        print(f"Label shape:", labels.shape)

        # Kiểm tra nếu feature quá nhỏ
        if features.shape[1] < min_length:
            print(f"[Warning] Feature có chiều dài {features.shape[1]}, nhỏ hơn {min_length}!")
            print("Có thể gây lỗi khi dùng MaxPooling1D. Cân nhắc padding hoặc kiểm tra pipeline trích xuất.")

            
            if save_fixed:
                max_length = min_length  
                print(f"[Fixing] Padding feature lên {max_length}...")
                features = np.array([
                    np.pad(f, (0, max_length - len(f))) if len(f) < max_length else f
                    for f in features
                ])

                # Lưu dữ liệu đã chỉnh sửa
                fixed_feature_path = feature_path.replace("train.p", "checked_train.p")
                joblib.dump((features, labels), fixed_feature_path)
                print(f"[INFO] Dữ liệu đã được lưu vào {fixed_feature_path}")

        # Kiểm tra và in ra toàn bộ nhãn
        unique_labels = np.unique(labels)  # Lấy các nhãn duy nhất
        print(f"[INFO] Các nhãn duy nhất: {unique_labels}")
        print(f"[INFO] Số lượng nhãn duy nhất: {len(unique_labels)}")

        # In ra toàn bộ nhãn
        print(f"[INFO] Tất cả các nhãn:")
        for label in unique_labels:
            print(label)

        return features.shape


    except Exception as e:
        print(f"[Error] Lỗi khi tải feature file: {e}")
        return None


if __name__ == "__main__":
    feature_shape = check_feature_size(save_fixed=True)

    if feature_shape:
        print(" Dữ liệu hợp lệ")
    else:
        print(" Dữ liệu có vấn đề")
