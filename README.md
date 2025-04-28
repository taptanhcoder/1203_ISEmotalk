# Speech Emotion Recognition


1203_ISEMOTALK/
├── config/                     # Lưu trữ file cấu hình json
│   ├── base_config.json        # Cấu hình chung
│   ├── cnn_config.json         # Cấu hình cho CNN

│   └── feature_extraction.json  # Cấu hình trích xuất đặc trưng
│
├── data/                        # Lưu trữ dữ l
│   ├── EMNS_data                # Dữ liệu gốc chưa qua xử lý
│   ├── RAVDESS                  # Dữ liệu gốc chưa qua xử lý
│   └── metadata.json            # Thông tin về dữ liệu
│
├── features/                    #  Lưu trữ đặc trưng đã trích xuất (quản lý bằng DVC)
│   ├── opensmile/               # Đặc trưng từ OpenSMILE
│   ├── librosa/                 # Đặc trưng từ Librosa
│   ├── wav2vec2/                # Đặc trưng từ Wav2Vec2
│   ├── train.csv                # Đặc trưng training
│   ├── test.csv                 # Đặc trưng testing
│   └── predict.csv              # Đặc trưng dự đoán
│
├── models/                      #  Chứa các mô hình
│   ├── __init__.py              # Quản lý import mô hình ( hàm khởi tạo tương tự như một construction)
│   ├── base_model.py            # Lớp cha cho các mô hình
│   ├── dnn                      # Mô hình DNN
        ├──__init__.py
        ├──dnn_control.py        # lớp cha định nghĩa cho các model dnn
        ├──cnn.py
│   ├── ml                       # Mô hình machine learning
│
├── extract_feats/               #  Trích xuất đặc trưng
│   ├── __init__.py              # Quản lý import
│   ├── opensmile_extractor.py   # Trích xuất bằng OpenSMILE
│   ├── librosa_extractor.py     # Trích xuất bằng Librosa
│   ├── wav2vec2_extractor.py    # Trích xuất bằng Wav2Vec2
│   └── feature_registry.py      # Đăng ký phương pháp trích xuất
│
├── utils/                       #  Các tiện ích hỗ trợ
│   ├── __init__.py              # Quản lý import
│   ├── file_utils.py            # Xử lý file, đổi tên, copy, move
│   ├── plot_utils.py            # Vẽ spectrogram, waveform, radar chart
│   ├── config_utils.py          # Load config từ YAML
│   ├── logging_utils.py         # Ghi log bằng TensorBoard / Weights & Biases
│   ├── evaluation_utils.py      # Đánh giá mô hình (Accuracy, F1-Score)
│   ├── dataset_utils.py         # Chia tập train/test, quản lý dữ liệu
│   ├── wandb_integration.py     # Tích hợp với Weights & Biases
│   └── dvc_utils.py             # Quản lý dữ liệu với DVC
│
├── train.py                     #  Train mô hình
├── predict.py                   #  Dự đoán cảm xúc từ file âm thanh
├── preprocess.py                #  Tiền xử lý dữ liệu
├── test.py                      #  Kiểm thử mô hình với tập test
│
├── checkpoints/                  #  Lưu trữ mô hình đã train (quản lý bằng MLflow)
│   ├── cnn/                      # Checkpoint của CNN
│   ├── lstm/                     # Checkpoint của LSTM
│   ├── transformer/              # Checkpoint của Transformer
│   └── model_metadata.json       # Metadata của các mô hình
│
├── notebooks/                    #  Chứa notebook Jupyter để phân tích dữ liệu
│   ├── data_analysis.ipynb       # Phân tích dữ liệu
│   ├── model_experiments.ipynb   # Thử nghiệm mô hình
│   └── feature_visualization.ipynb # Trực quan hóa đặc trưng
│
├── scripts/                      #  Chứa các script hỗ trợ
│   ├── convert_audio.py          # Chuyển đổi định dạng audio
│   ├── run_training.sh           # Chạy training bằng shell script
│   ├── deploy_model.py           # Triển khai mô hình
│   ├── clean_dataset.py          # Dọn dẹp dữ liệu
│   └── inference_benchmark.py    # Benchmark inference time
│
├── requirements.txt              #  Danh sách thư viện cần cài đặt
├── README.md                     #  Hướng dẫn sử dụng dự án
├── .gitignore                    #  Bỏ qua file không cần push lên GitHub
└── .dvc/                         #  Quản lý version dataset với DVC


