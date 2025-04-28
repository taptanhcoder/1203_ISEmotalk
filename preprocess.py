import extract_feature as ef
from utils import parse_opt

if __name__ == '__main__':
    config = parse_opt()

    data_path = ef.get_path(config.dataset.data_path, train=True)
    ef.get_features(data_path,config,train=True)

    