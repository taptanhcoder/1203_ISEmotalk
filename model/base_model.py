import numpy as np
from abc import ABC, abstractmethod
from typing import Union 
from tensorflow.keras.models import Sequential # type: ignore
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class BaseModel(ABC):
    """
    Lớp trừu tượng (Abstract Class) cho mô hình Deep Learning.
    """

    def __init__(
        self,
        model: Union[Sequential],  
        trained: bool = False  
    ) -> None:
        """
        Khởi tạo mô hình.

        Args:
            model (Sequential): Mô hình Keras.
            trained (bool): Trạng thái mô hình 
        """
        self.model = model
        self.trained = trained

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Huấn luyện mô hình trên dữ liệu huấn luyện.
        
        Args:
            x_train (np.ndarray): Dữ liệu đầu vào để train.
            y_train (np.ndarray): Nhãn tương ứng với dữ liệu train.
        """
        pass


    @abstractmethod
    def predict(self, sample: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn của dữ liệu đầu vào.

        Args:
            sample (np.ndarray): Dữ liệu cần dự đoán.

        Returns:
            np.ndarray: Kết quả dự đoán.
        """
        pass

 

    @abstractmethod
    def save(self, path: str, name: str, config=None) -> None:
        """
        Lưu mô hình vào file.

        Args:
            path (str): Đường dẫn thư mục lưu file.
            name (str): Tên file lưu mô hình.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, name: str):
        """
        Tải mô hình từ file.

        Args:
            path (str): Đường dẫn thư mục chứa file.
            name (str): Tên file mô hình.

        Returns:
            BaseModel: Đối tượng mô hình đã tải.
        """
        pass

    @classmethod
    @abstractmethod
    def make(cls):
        """
        Khởi tạo mô hình mới.

        Returns:
            BaseModel: Đối tượng mô hình mới.
        """
        pass



    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Đánh giá mô hình bằng độ chính xác trên tập kiểm tra.

        Args:
            x_test (np.ndarray): Dữ liệu kiểm tra.
            y_test (np.ndarray): Nhãn thực tế của tập kiểm tra.

        Returns:
            float: Giá trị độ chính xác của mô hình.
        """
        prediction = self.predict(x_test)  
        
        
        
        if prediction.ndim > 1 and prediction.shape[1] > 1:
            prediction = np.argmax(prediction, axis=1)

        if y_test.ndim > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test, prediction)  
        print('Accuracy: %.3f\n' % accuracy)
        return accuracy



