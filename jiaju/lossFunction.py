import torch
import numpy as np
import torch.nn.functional as func
from torch import nn,optim
import os
class privacyLoss(nn.Module):
    def __init__(self,cov, batch_size,device):
        super(privacyLoss,self).__init__()
        self.device = device
        self.batchSize = batch_size
        self.par = (cov*0.5/batch_size).to(device=device)
        #直接得到1/(2*sigma*N)

    def forward(self, out, label):
        outs = torch.split(out,1,dim=0)
        lshape = label.shape
        lx = lshape[0]
        result = torch.Tensor([0.0]).float().to(device=self.device)

        for i in range(self.lx-1):
            for j in range(1,lx):
                if torch.equal(label[i], label[j]):
                    result += torch.pow(torch.norm((outs[i]-outs[j]).float(),2),2)
                else:
                    result -= torch.pow(torch.norm((outs[i]-outs[j]).float(),2),2)

        return (self.par*result).sum()

