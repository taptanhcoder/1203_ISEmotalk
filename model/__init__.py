from .dnn import CNN, DNN



_MODELS={
    'cnn': CNN
}


def make_model(config, n_feats:int,num_classes:int):
    """
    Tạo mô hình dựa trên cấu hình được cung cấp.
    """
    if config.model == 'cnn':
        model = CNN.make(config, n_feats, num_classes)  
    return model

def load(config):
    """
    Tải mô hình đã được huấn luyện từ file checkpoint.

    Args:
        config: Đối tượng chứa các tham số cấu hình, bao gồm đường dẫn checkpoint.

    Returns:
        model: Mô hình đã tải từ file.
    """
    return _MODELS[config.model].load(
        path=config.checkpoint.checkpoint_path,
        name=config.checkpoint.checkpoint_name
    )