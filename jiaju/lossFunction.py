import torch
from torch import nn

class privacyLoss1(nn.Module):
    def __init__(self,sigma,device):
        super(privacyLoss1,self).__init__()
        self.device = device
        self.sigma = sigma
        self.par = sigma*0.5
        #直接得到1/(2*sigma*N)

    def forward(self, feature, label):
        features = torch.split(feature, 1, dim=0)  # 得到的高斯分布
        batchSize = label.shape[0]  # batchSize
        par = self.par / batchSize  # 参数
        result = torch.Tensor([0.0]).float().to(device=self.device)

        for i in range(batchSize - 1):
            for j in range(1, batchSize):
                if torch.equal(label[i], label[j]):
                    result += torch.pow(torch.norm((features[i] - features[j]).float(), 2), 2)
                else:
                    result -= torch.pow(torch.norm((features[i] - features[j]).float(), 2), 2)
        return par * result


class privacyLoss2(nn.Module):
    def __init__(self, sigma, device):
        super(privacyLoss2, self).__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, feature, label):
        features = torch.split(feature, 1, dim=0)
        batchSize = label.shape[0]  # batchSize
        k = features[0].shape[1]#dimension

        temp_f1,temp_f2=0,0#count

        mu_f1 = torch.zeros((k,1)).float().to(device=self.device)
        mu_f2 = torch.zeros((k,1)).float().to(device=self.device)
        Sigma_f1 = torch.zeros((k,k)).float().to(device=self.device)
        Sigma_f2 = torch.zeros((k,k)).float().to(device=self.device)
        Sigma_a = torch.eye(k).float().to(device=self.device)
        for i in range(batchSize):
            if label[i]==0:
                mu_f1+=features[i].t()
                temp_f1+=1
            else:
                mu_f2+=features[i].t()
                temp_f2+=1
        mu_f1,mu_f2 = (mu_f1/temp_f1),(mu_f2/temp_f2)
        for i in range(batchSize):
            if label[i]==0:
                Sigma_f1+=torch.mm((features[i].t()-mu_f1),(features[i].t()-mu_f1).t())
            else:
                Sigma_f2+=torch.mm((features[i].t()-mu_f1),(features[i].t()-mu_f1).t())

        Sigma_f1,Sigma_f2 = Sigma_a+Sigma_f1/temp_f1,Sigma_a+Sigma_f2/temp_f2

        temp1 = (mu_f1-mu_f2)
        result = 0.5*(torch.log2(Sigma_f2.det()/Sigma_f1.det())-k
                      +torch.mm(torch.mm(temp1.t(),Sigma_f2.inverse()),temp1)
                      +torch.trace(torch.mm(Sigma_f2.inverse(),Sigma_f1)))

        return result
