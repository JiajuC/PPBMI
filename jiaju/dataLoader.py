import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
def load_adult_dataset(prvacyAtt,flag=0):
    dataPath = 'Data/adult/'
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
    if flag:
        X_train = np.delete(X_train,prvacyAtt,axis=1)

        X_test = np.delete(X_test,prvacyAtt,axis=1)


    train_data = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train)
    test_data = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test)

    return train_data,train_label,test_data,test_label


def construct_data_loader(train_data,train_label,test_data,test_label,batch_size):
    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_data_loader(dataset_name,prvacyAtt,flag=0):
    if dataset_name=='adult':
        train_data,train_label,test_data,test_label = load_adult_dataset(prvacyAtt,flag)
    train_loader,test_loader = construct_data_loader(train_data,train_label,test_data,test_label,32)
    return train_loader,test_loader