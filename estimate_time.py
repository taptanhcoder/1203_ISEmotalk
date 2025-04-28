import os
import time
import numpy as np
import extract_feature as ef
import model
import utils
from tensorflow.keras.models import load_model

def predict(config, audio_path: str) -> None:
    """
    Dự đoán nhãn của file âm thanh và đo thời gian từng giai đoạn.

    Args:
        config: Đối tượng cấu hình chứa thông tin về mô hình và đường dẫn.
        audio_path (str): Đường dẫn file âm thanh cần dự đoán.
    """

    total_start_time = time.time()  
    audio_path = os.path.abspath(audio_path)


    path_start_time = time.time()
    paths_list = ef.get_path(audio_path, train=False)
    path_time = time.time() - path_start_time
    print(f"Thời gian lấy đường dẫn file: {path_time:.4f} giây")


    feature_start_time = time.time()
    test_feature = ef.get_features(paths_list, config, train=False)
    feature_time = time.time() - feature_start_time
    print(f"Thời gian trích xuất đặc trưng: {feature_time:.4f} giây")


    model_start_time = time.time()
    checkpoint_path = os.path.join(config.checkpoint.checkpoint_path, config.checkpoint.checkpoint_name + ".keras")
    model = load_model(checkpoint_path)
    model_load_time = time.time() - model_start_time
    print(f"Thời gian tải mô hình: {model_load_time:.4f} giây")


    predict_start_time = time.time()
    result = model.predict(test_feature, verbose=0)  
    predict_time = time.time() - predict_start_time
    print(f"Thời gian dự đoán: {predict_time:.4f} giây")


    class_labels = config.dataset.class_labels
    predicted_class = np.argmax(result, axis=1)  
    print(f"Nhãn dự đoán: {class_labels[predicted_class[0]]}")


    total_time = time.time() - total_start_time
    print(f"Tổng thời gian thực hiện: {total_time:.4f} giây")

if __name__ == '__main__':
    audio_path = "data/predict_data/Sad (2).wav"
    config = utils.parse_opt()
    predict(config, audio_path)
