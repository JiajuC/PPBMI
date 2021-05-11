import torch
from torch import nn
import datetime
class privacyLoss1(nn.Module):
    def __init__(self,sigma,device):
        super(privacyLoss1,self).__init__()
        self.device = device
        self.sigma = sigma
        self.par = sigma*0.5
        #直接得到1/(2*sigma*N)

    def forward(self, feature, privatelabel,pan):

        features = torch.split(feature, 1, dim=0)  # 得到的高斯分布
        batchSize = privatelabel.shape[0]  # batchSize
        par = self.par / batchSize  # 参数
        result = torch.Tensor([0.0]).float().to(device=self.device)
        #0.002
        for i in range(batchSize - 1):
            for j in range(1, batchSize):
                if torch.equal(privatelabel[i], privatelabel[j]):
                    result -= torch.pow(torch.norm((features[i] - features[j]).float(), 2), 2)
                else:
                    result += torch.pow(torch.norm((features[i] - features[j]).float(), 2), 2)
        #0.15左右

        return par * result

class privacyLoss2(nn.Module):
    def __init__(self, sigma, device):
        super(privacyLoss2, self).__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, feature, privatelabel, pan):

        features = torch.split(feature, 1, dim=0)#32*16的feature，
        batchSize = privatelabel.shape[0]  # batchSize
        k = features[0].shape[1]  # dimension
        type_num = [9, 4, 9, 7, 15]
        keys = []#用来进行索引的，存储的是以下三个字典的key,key是隐私标签名称
        mu_Fs = {}#不同隐私类的mu
        temp_fs = {}#存储的是一个batch中有多少个隐私类
        Sigma_Fs = {}#不同隐私类的sigma
        Sigma_a = torch.eye(k).float().to(device=self.device)

        #求方差
        for i in range(batchSize):
            label = float(privatelabel[i])
            if label in mu_Fs:
                mu_Fs[label] += features[i].t()
                temp_fs[label] += 1
            else:
                mu_Fs[label] = features[i].t()
                temp_fs[label] = 1

        #求均值
        for i in range(batchSize):
            label = float(privatelabel[i])
            mu_f = mu_Fs[label] / temp_fs[label]
            if label in Sigma_Fs:
                Sigma_Fs[label] += torch.mm((features[i].t() - mu_f), (features[i].t() - mu_f).t())
            else:
                Sigma_Fs[label] = torch.mm((features[i].t() - mu_f), (features[i].t() - mu_f).t())

        result = torch.Tensor([0.0]).float().to(device=self.device)


        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / temp_fs[key]
            Sigma_Fs[key] = Sigma_a + Sigma_Fs[key] / temp_fs[key]
            keys.append(key)
            result -= torch.mm(mu_Fs[key].t(), mu_Fs[key])[0]#先减去方差平方的和
        #0.02

        #然后再加上两两之间的kldiv 这一步最耗时
        for i in range(len(keys) - 1):
            s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            for j in range(1, len(keys)):
                s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                result += self.kldiv(s1, u1, s2, u2, k)[0]
        #0.4-0.8

        return result

    def kldiv(self,s1, u1, s2, u2, k):
        temp1 = (u1 - u2)
        result = 0.5 * (torch.log2(s2.det() / s1.det()) - k
                        + torch.mm(torch.mm(temp1.t(), s2.inverse()), temp1)
                        + torch.trace(torch.mm(s2.inverse(), s1)))
        return result

class privacyLoss3(nn.Module):
    def __init__(self, sigma, device):
        super(privacyLoss3, self).__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, feature, label,pan):
        features = torch.split(feature, 1, dim=0)
        batchSize = label.shape[0]  # batchSize
        k = features[0].shape[1]#dimension

        temp_f1,temp_f2=0,0#count
        print(label)
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
