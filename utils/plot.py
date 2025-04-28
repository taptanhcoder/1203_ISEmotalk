import wave
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wav
import numpy as np
import os

def curve(train: list, val: list, title: str, y_label: str, save_path: str = "utils/plots/") -> None:
    """
   Vẽ đồ thị loss và accuracy và lưu vào file thay vì hiển thị.

    Args:
        train (list): Danh sách loss/accuracy của tập train.
        val (list): Danh sách loss/accuracy của tập validation.
        title (str): Tiêu đề đồ thị.
        y_label (str): Nhãn trục Y.
        save_path (str, optional): Thư mục lưu ảnh, mặc định là "plots/".
    """
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    
    os.makedirs(save_path, exist_ok=True)
    
    file_path = os.path.join(save_path, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(file_path)  
    print(f"Saved plot at {file_path}")
    
    plt.close()  




def radar(data_prob: np.ndarray, class_labels: list, save_path: str = "utils/plots/") -> None:
    """
    Vẽ biểu đồ radar xác suất cảm xúc và lưu vào file.

    Args:
        data_prob (np.ndarray): Mảng xác suất của từng class.
        class_labels (list): Danh sách các nhãn cảm xúc.
        save_path (str, optional): Thư mục lưu ảnh, mặc định là "utils/plots/".
    """

    data_prob = np.squeeze(data_prob)
    

    if len(data_prob) != len(class_labels):
        print(f" Warning: Số class ({len(class_labels)}) không khớp với dữ liệu ({len(data_prob)})")
        return

    angles = np.linspace(0, 2 * np.pi, len(class_labels), endpoint=False)

    data = np.concatenate((data_prob, [data_prob[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={"projection": "polar"})
    
    ax.plot(angles, data, "o-", linewidth=2, label="Xác suất dự đoán")
    ax.fill(angles, data, alpha=0.3)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, class_labels)
    ax.set_title("Dự đoán Cảm Xúc", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.6)


    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "radar_chart.png")
    plt.savefig(file_path, dpi=300)
    print(f" Saved radar chart at {file_path}")

    plt.close()