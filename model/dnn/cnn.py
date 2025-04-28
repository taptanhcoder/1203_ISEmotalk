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
from .dnn_control import DNN
from ..base_model import BaseModel  

class CNN(DNN, ABC):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        """
        Khởi tạo mô hình CNN.

        Tham số:
            model (Sequential): Mô hình Keras Sequential.
            trained (bool): Trạng thái mô hình, xác định xem mô hình đã được huấn luyện hay chưa.
        """
        super(CNN, self).__init__(model, trained)
        print(self.model.summary())


    @classmethod
    def make(cls, config,n_feats:int,num_classes:int ):
        """
        Tạo mô hình CNN từ object Config.

        Tham số:
            config (Config): Object chứa thông số cấu hình.

        Trả về:
            CNN: Một instance của lớp CNN.
        """
        
        

        model = Sequential()

        # Thêm các lớp Conv1D
        for i, layer in enumerate(config.model_params.conv_layers):
            if i == 0:
                model.add(Conv1D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides_1,
                    padding=layer.padding,
                    activation=layer.activation,
                    input_shape=(n_feats,1)
                ))
            else:
                model.add(Conv1D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides_1,
                    padding=layer.padding,
                    activation=layer.activation
                ))

            
            if layer.batch_norm:
                model.add(BatchNormalization())

            # Thêm MaxPooling1D
            model.add(MaxPooling1D(
                pool_size=layer.pool_size,
                strides=layer.strides_2
            ))

            # Kiểm tra dropout trước khi thêm
            if layer.dropout > 0:
                model.add(Dropout(layer.dropout))

        # Flatten layer
        model.add(Flatten())

        # Thêm các lớp Dense
        for layer in config.model_params.dense_layers:
            model.add(Dense(layer.units, activation=layer.activation))

            if layer.batch_norm:
                model.add(BatchNormalization())

            if layer.dropout > 0:
                model.add(Dropout(layer.dropout))

        # Lớp output
        model.add(Dense(num_classes, activation=config.model_params.output_layer.activation))

        

        return cls(model)
    

    


    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """
        Định hình lại dữ liệu đầu vào theo định dạng yêu cầu.

        Tham số:
            data (np.ndarray): Dữ liệu đầu vào.

        Trả về:
            np.ndarray: Dữ liệu đầu vào đã được định hình lại.
        """
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        return data