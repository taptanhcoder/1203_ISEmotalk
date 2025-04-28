import os
import json
from typing import Optional
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping # type: ignore
from keras.models import Sequential, model_from_json # type: ignore
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization # type: ignore
from keras.utils import to_categorical # type: ignore
from abc import ABC, abstractmethod
from utils import curve
from ..base_model import BaseModel  
from tensorflow.keras.models import load_model

class DNN(BaseModel, ABC):

    def __init__(self, model: Sequential, trained: bool = False) -> None:
        """
        Khởi tạo mô hình DNN.

        Tham số:
            model (Sequential): Mô hình Keras Sequential.
            trained (bool): Trạng thái mô hình đã được huấn luyện hay chưa.
        """
        super(DNN, self).__init__(model, trained)
        print(self.model.summary())



    @classmethod
    def load(cls, path: str, name: str):
        """
        Tải mô hình từ file .keras.

        Tham số:
            path (str): Đường dẫn thư mục chứa file mô hình.
            name (str): Tên file mô hình.

        Trả về:
            DNN: Instance của mô hình đã tải.
        """
        model_path = os.path.join(path, name + ".keras")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} không tồn tại!")


        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")

        return cls(model, trained=True)



    def save(self, path: str, name: str, config=None) -> None:
        """
        Lưu toàn bộ mô hình với trọng số tốt nhất (nếu có).

        Tham số:
            path (str): Thư mục lưu mô hình.
            name (str): Tên file lưu mô hình.
            config (dict): Cấu hình lưu trọng số tốt nhất.
        """
        os.makedirs(path, exist_ok=True)

    
        best_model_path = os.path.join(config.callbacks.model_checkpoint.path)

        if os.path.exists(best_model_path):
            self.model = load_model(best_model_path)  
            print(f"Loaded best model from {best_model_path}")
        else:
            print(f"Warning: No best model found at {best_model_path}. Saving the current model instead.")

        
        final_model_path = os.path.join(path, name + ".keras")
        self.model.save(final_model_path)  
        print(f"Model saved to {final_model_path}")

            

    def train(self, x_train, y_train, x_val=None, y_val=None, config=None) -> None:
        """
        Huấn luyện mô hình CNN với config từ `parse_opt()`.

        Tham số:
            x_train (np.ndarray): Dữ liệu huấn luyện.
            y_train (np.ndarray): Nhãn huấn luyện.
            x_val (np.ndarray, tùy chọn): Dữ liệu validation (mặc định là None).
            y_val (np.ndarray, tùy chọn): Nhãn validation (mặc định là None).
            config (Config): Object chứa cấu hình (nhận từ `parse_opt()`).
        """
        if config is None:
            raise ValueError("Config không được để trống. Hãy truyền vào config từ parse_opt().")


        training_config = config.training
        callbacks_config = config.callbacks

        batch_size = training_config.batch_size
        n_epochs = training_config.epochs

        
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        
        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)

        
        callbacks = []
    
        if hasattr(callbacks_config, "model_checkpoint"):
            checkpoint = ModelCheckpoint(
                filepath=callbacks_config.model_checkpoint.path,
                monitor=callbacks_config.model_checkpoint.monitor,
                save_best_only=callbacks_config.model_checkpoint.save_best_only,
                verbose=1
            )
            callbacks.append(checkpoint)

        if hasattr(callbacks_config, "early_stopping"):
            early_stop = EarlyStopping(
                monitor=callbacks_config.early_stopping.monitor,
                patience=callbacks_config.early_stopping.patience,
                restore_best_weights=callbacks_config.early_stopping.restore_best_weights,
                verbose=1
            )
            callbacks.append(early_stop)

        if hasattr(callbacks_config, "reduce_lr"):
            reduce_lr = ReduceLROnPlateau(
                monitor=callbacks_config.reduce_lr.monitor,
                patience=callbacks_config.reduce_lr.patience,
                factor=callbacks_config.reduce_lr.factor,
                min_lr=callbacks_config.reduce_lr.min_lr,
                verbose=1
            )
            callbacks.append(reduce_lr)

        self.model.compile(
            optimizer='adam',  
            loss='categorical_crossentropy',  
            metrics=['accuracy']  
        )

        # Huấn luyện mô hình
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_val, y_val),
            shuffle=True,
            verbose=1,
            callbacks=callbacks
        )

        # Lấy dữ liệu huấn luyện
        acc = history.history.get("accuracy", [])
        loss = history.history.get("loss", [])
        val_acc = history.history.get("val_accuracy", [])
        val_loss = history.history.get("val_loss", [])

        # Vẽ đồ thị loss/accuracy nếu có
        if acc and val_acc:
            curve(acc, val_acc, "Accuracy", "acc")
        if loss and val_loss:
            curve(loss, val_loss, "Loss", "loss")

        self.trained = True



    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn lớp cho dữ liệu đầu vào.

        Tham số:
            x (np.ndarray): Dữ liệu đầu vào.

        Trả về:
            np.ndarray: Nhãn dự đoán.
        """
        if not self.trained:
            raise RuntimeError("Chưa có mô hình nào được huấn luyện.")

        samples = self.reshape_input(x)
        return np.argmax(self.model.predict(samples), axis=1)
    



    def predict_proba(self, sample: np.ndarray) -> np.ndarray:
        """
        Dự đoán phần trăm xác suất của từng lớp.

        Tham số:
            sample (np.ndarray): Dữ liệu đầu vào.

        Trả về:
            np.ndarray: Mảng xác suất của từng lớp.
        """

        sample = self.reshape_input(sample)


        if sample.ndim == 1:
            sample = np.expand_dims(sample, axis=0) 


        probabilities = self.model.predict(sample, verbose=0)  

        return np.squeeze(probabilities)
    


    @abstractmethod
    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """
        Định hình lại dữ liệu đầu vào theo định dạng yêu cầu.

        Tham số:
            data (np.ndarray): Dữ liệu đầu vào.

        Trả về:
            np.ndarray: Dữ liệu đã được định hình lại.
        """
        pass