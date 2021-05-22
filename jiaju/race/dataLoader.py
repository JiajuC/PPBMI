import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
def load_adult_dataset():

    dataPath = r'data/'
    f1 = open(dataPath + r'race_train_data.pickle', 'rb')
    f2 = open(dataPath + r'race_train_label.pickle', 'rb')
    f3 = open(dataPath + r'race_test_data.pickle', 'rb')
    f4 = open(dataPath + r'race_test_label.pickle', 'rb')
    f5 = open(dataPath + r'race_train_privacy.pickle', 'rb')
    f6 = open(dataPath + r'race_test_privacy.pickle', 'rb')
    X_train = np.array(pickle.load(f1), dtype='float32')
    y_train = pickle.load(f2)
    X_test = np.array(pickle.load(f3), dtype='float32')
    y_test = pickle.load(f4)
    z_train = pickle.load(f5)
    z_test = pickle.load(f6)
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()

    train_data = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train)
    test_data = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test)
    train_privact_att = torch.from_numpy(z_train)
    test_privact_att = torch.from_numpy(z_test)

    return train_data, train_label, test_data, test_label, train_privact_att, test_privact_att

def construct_data_loader(train_data,train_label,test_data,test_label,train_privact_att, test_privact_att,batch_size):
    train_dataset = TensorDataset(train_data, train_label,train_privact_att)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_data, test_label, test_privact_att)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def construct_data_loader1(train_data,train_label,test_data,test_label,batch_size):
    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
