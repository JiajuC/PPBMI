import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import datetime
import time
from tqdm import tqdm
features = 16
cov = torch.eye(features)#Covariance matrix which is initialized to one
batchSize = 32
print("%d"%1e-3)