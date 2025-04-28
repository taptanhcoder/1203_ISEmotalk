#  Speech Emotion Recognition - 1203_ISEMOTALK

Dự án nhận diện cảm xúc từ giọng nói sử dụng các kỹ thuật Deep Learning, Machine Learning và các công cụ như Librosa, OpenSMILE, Wav2Vec2, DVC, MLflow.

## Cấu Trúc Thư Mục Dự Án

```plaintext
1203_ISEMOTALK/
├── config/                     # Lưu trữ file cấu hình json
│   ├── base_config.json        # Cấu hình chung
│   ├── cnn_config.json         # Cấu hình cho CNN
│   └── feature_extraction.json # Cấu hình trích xuất đặc trưng
│
├── data/                       # Lưu trữ dữ liệu gốc
│   ├── EMNS_data
│   ├── RAVDESS
│   └── metadata.json
│
├── features/                   # Đặc trưng đã trích xuất (quản lý bằng DVC)
│   ├── opensmile/
│   ├── librosa/
│   ├── wav2vec2/
│   ├── train.csv
│   ├── test.csv
│   └── predict.csv
│
├── models/                     # Chứa các mô hình
│   ├── __init__.py
│   ├── base_model.py
│   ├── dnn/
│   │   ├── __init__.py
│   │   ├── dnn_control.py
│   │   └── cnn.py
│   └── ml/
│
├── extract_feats/              # Trích xuất đặc trưng
│   ├── __init__.py
│   ├── opensmile_extractor.py
│   ├── librosa_extractor.py
│   ├── wav2vec2_extractor.py
│   └── feature_registry.py
│
├── utils/                      # Các tiện ích hỗ trợ
│   ├── __init__.py
│   ├── file_utils.py
│   ├── plot_utils.py
│   ├── config_utils.py
│   ├── logging_utils.py
│   ├── evaluation_utils.py
│   ├── dataset_utils.py
│   ├── wandb_integration.py
│   └── dvc_utils.py
│
├── train.py                    # Train mô hình
├── predict.py                  # Dự đoán cảm xúc
├── preprocess.py               # Tiền xử lý dữ liệu
├── test.py                     # Kiểm thử mô hình
│
├── checkpoints/                # Lưu trữ mô hình đã train (quản lý bằng MLflow)
│   ├── cnn/
│   ├── lstm/
│   ├── transformer/
│   └── model_metadata.json
│
├── notebooks/                  # Notebook Jupyter
│   ├── data_analysis.ipynb
│   ├── model_experiments.ipynb
│   └── feature_visualization.ipynb
│
├── scripts/                    # Script hỗ trợ
│   ├── convert_audio.py
│   ├── run_training.sh
│   ├── deploy_model.py
│   ├── clean_dataset.py
│   └── inference_benchmark.py
│
├── requirements.txt            # Danh sách thư viện
├── README.md                   # Hướng dẫn dự án
├── .gitignore                  # File Git ignore
└── .dvc/                       # Quản lý version dataset

