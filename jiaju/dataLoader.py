import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
def load_adult_dataset(prvacyAtt,flag=0,dataSet_type = 1):
    dataPath = r'Data/'
    if dataSet_type == 1:
        f1 = open(dataPath + r'adult_Income/train_data.pickle', 'rb')
        f2 = open(dataPath + r'adult_Income/train_label.pickle', 'rb')
        f3 = open(dataPath + r'adult_Income/test_data.pickle', 'rb')
        f4 = open(dataPath + r'adult_Income/test_label.pickle', 'rb')
    else:
        f1 = open(dataPath + r'adult_workClass/workClass_train_data.pickle', 'rb')
        f2 = open(dataPath + r'adult_workClass/workClass_train_label.pickle', 'rb')
        f3 = open(dataPath + r'adult_workClass/workClass_test_data.pickle', 'rb')
        f4 = open(dataPath + r'adult_workClass/workClass_test_label.pickle', 'rb')
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


def get_data_loader(dataset_name,prvacyAtt,flag=0,type=1):
    if dataset_name=='adult':
        train_data,train_label,test_data,test_label = load_adult_dataset(prvacyAtt,flag,type)
    train_loader,test_loader = construct_data_loader(train_data,train_label,test_data,test_label,32)
    return train_loader,test_loader