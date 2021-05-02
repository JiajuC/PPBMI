import torch
from torch import nn,optim

class privacyLoss1(nn.Module):
    def __init__(self,sigma,device):
        super(privacyLoss1,self).__init__()
        self.device = device
        self.sigma = sigma
        self.par = sigma*0.5
        #直接得到1/(2*sigma*N)

    def forward(self, out, label):
        outs = torch.split(out,1,dim=0)
        lx = label.shape[0]
        par = self.par/lx
        result = torch.Tensor([0.0]).float().to(device=self.device)

        for i in range(lx-1):
            for j in range(1,lx):
                if torch.equal(label[i], label[j]):
                    result += torch.pow(torch.norm((outs[i]-outs[j]).float(),2),2)
                else:
                    result -= torch.pow(torch.norm((outs[i]-outs[j]).float(),2),2)
        return par*result

