import argparse
import json

class Config:
    """dict -> Class"""
    def __init__(self, entries: dict = {}):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(v)
            elif isinstance(v, list) and all(isinstance(i, dict) for i in v): 
                self.__dict__[k] = [Config(i) for i in v]  
            else:
                self.__dict__[k] = v


def load_config(file_path: str) -> dict:
    """
    Tải cấu hình từ file JSON

    Args:
        file_path (str): Đường dẫn file cấu hình JSON

    Returns:
        config (dict): Dữ liệu cấu hình dưới dạng dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def parse_opt():
    """
    Xử lý tham số dòng lệnh để nhận đường dẫn file cấu hình JSON
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/cnn.json',
        help='Đường dẫn tới file cấu hình JSON'
    )
    args = parser.parse_args()
    config_dict = load_config(args.config)  
    config = Config(config_dict)  
    
    return config
