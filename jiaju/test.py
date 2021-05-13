import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import datetime
features = 16
cov = torch.eye(features)#Covariance matrix which is initialized to one
batchSize = 32

a = torch.rand(5,3)
print(a)
a=a[:,0]
print(float(a[0]))
def forward(self, feature, privatelabel,pan):
    features = torch.split(feature, 1, dim=0)
    batchSize = privatelabel.shape[0]  # batchSize
    k = features[0].shape[1]  # dimension
    type_num = [9, 4, 9, 7, 15]
    keys = []
    mu_Fs = {}
    temp_fs = {}
    Sigma_Fs = {}
    Sigma_a = torch.eye(k).float().to(device=self.device)
    for i in range(batchSize):
        label = float(privatelabel[i])
        if label in mu_Fs:
            mu_Fs[label]+=features[i].t()
            temp_fs[label]+=1
        else:
            mu_Fs[label] = features[i].t()
            temp_fs[label] = 1
    for i in range(batchSize):
        label = float(privatelabel[i])
        mu_f = mu_Fs[label] / temp_fs[label]
        if label in Sigma_Fs:
            Sigma_Fs[label]+= torch.mm((features[i].t()-mu_f),(features[i].t()-mu_f).t())
        else:
            Sigma_Fs[label] = torch.mm((features[i].t() - mu_f), (features[i].t() - mu_f).t())

    result = torch.Tensor([0.0]).float().to(device=self.device)
    for key in mu_Fs:
        mu_Fs[key] = mu_Fs[key]/temp_fs[key]
        Sigma_Fs[key] = Sigma_a+Sigma_Fs[key]/temp_fs[key]
        keys.append(key)
        result-=torch.mm(mu_Fs[key],mu_Fs[key].t())[0]

    for i in range(len(keys)-1):
        s1,u1 = Sigma_Fs[keys[i]],mu_Fs[keys[i]]
        for j in range(1,len(keys)):
            s2,u2 = Sigma_Fs[keys[j]],mu_Fs[keys[j]]
            kltime = datetime.datetime.now()
            result+=kldiv(s1,u1,s2,u2,k)
            kltime = datetime.datetime.now()-kltime
    return result
def kldiv(s1,u1,s2,u2,k):
    temp1 = (u1 - u2)
    result = 0.5 * (torch.log2(s2.det() / s1.det()) - k
                    + torch.mm(torch.mm(temp1.t(), s2.inverse()), temp1)
                    + torch.trace(torch.mm(s2.inverse(), s1)))
    return result
