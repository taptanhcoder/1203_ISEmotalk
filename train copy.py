import extract_feature as ef
from tensorflow.keras.utils import to_categorical
import model
from utils import parse_opt
import numpy as np


def train(config)->None:
    
    data_path = ef.get_path(config.dataset.data_path, train=True)
    x_train, x_test, y_train, y_test=ef.get_features(data_path,config,train=True)

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    num_classes = len(np.unique(y_train))  
    train_model = model.make_model(config=config, n_feats=x_train.shape[1], num_classes=num_classes)

    print('----- start training', config.model, '-----')

    if config.model in['cnn']:
        y_train, y_test = to_categorical(y_train), to_categorical(y_test)  
        train_model.train(
            x_train, y_train,
            x_test, y_test,
            config
        )
    else:
        train_model.train(x_train, y_train)
        
    print('----- end training ', config.model, ' -----')

    train_model.evaluate(x_test,y_test)
    train_model.save(config.checkpoint.checkpoint_path, config.checkpoint.checkpoint_name, config)
     

if __name__ == '__main__':
    config = parse_opt()
    train(config)



