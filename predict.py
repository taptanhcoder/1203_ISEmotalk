import os 
import numpy as np
import extract_feature as ef
import model
import utils
from tensorflow.keras.models import load_model

def predict(config, audio_path: str) -> None:
    """
    Dự đoán nhãn của file âm thanh.

    Args:
        config: Đối tượng cấu hình chứa thông tin về mô hình và đường dẫn.
        audio_path (str): Đường dẫn file âm thanh cần dự đoán.
    """

    audio_path = os.path.abspath(audio_path)

    paths_list = ef.get_path(audio_path, train=False)


    test_feature = ef.get_features(paths_list, config, train=False)

    checkpoint_path = os.path.join(config.checkpoint.checkpoint_path, config.checkpoint.checkpoint_name + ".keras")
    model = load_model(checkpoint_path)
    print(f" Model loaded from {checkpoint_path}")

    result = model.predict(test_feature, verbose=0)  

    predicted_class = np.argmax(result, axis=1)  

    class_labels = config.dataset.class_labels

    print(f"Nhãn dự đoán: {class_labels[predicted_class[0]]}") 


if __name__ == '__main__':
    audio_path = "data/EMNS_Data/recorded_audio_0UbW71l.wav"
    config = utils.parse_opt()
    predict(config, audio_path)