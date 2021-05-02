import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
from lossFunction import privacyLoss
import os

features = 16
cov = torch.eye(features)#Covariance matrix which is initialized to one
batchSize = 32

def getGaussian(self, feature):
    # feature1 = feature.cpu().detach().numpy()
    temp = np.random.multivariate_normal([0 for i in range(features)], self.cov)
    # Data are generated according to covariance matrix and mean
    for i in range(1, batchSize):
        temp = np.concatenate((temp, np.random.multivariate_normal([0 for i in range(feature1[0])],
                                                                   self.cov)),
                              axis=0)
        # Splicing sampling of high dimensional Gaussian distribution data
    temp.resize((features, features))
    # Since the stitched data is one-dimensional,
    # we redefine it as the original dimension
    temp = torch.from_numpy(temp).float()
    feature = feature+temp.to(self.device)+feature
    return feature

loss = torch.Tensor([[1,2]])
print(loss.shape[1])