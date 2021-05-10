import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_adult_dataset():
    dataPath = '../data/adult/'
    f1 = open(dataPath + 'train_data.pickle', 'rb')
    f2 = open(dataPath + 'train_label.pickle', 'rb')
    f3 = open(dataPath + 'test_data.pickle', 'rb')
    f4 = open(dataPath + 'test_label.pickle', 'rb')
    X_train = np.array(pickle.load(f1), dtype='float32')
    y_train = pickle.load(f2)
    X_test = np.array(pickle.load(f3), dtype='float32')
    y_test = pickle.load(f4)
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    train_data = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train)
    test_data = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test)

    return train_data,train_label,test_data,test_label


def construct_data_loader(data,label,batch_size):
    dataset = TensorDataset(data, label)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def get_data_loader(dataset_name):
    if dataset_name=='adult':
        train_data,train_label,test_data,test_label = load_adult_dataset()
    train_loader = construct_data_loader(train_data,train_label,32)
    test_loader = construct_data_loader(test_data,test_label,32)
    return train_loader,test_loader