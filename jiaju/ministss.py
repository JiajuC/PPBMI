import numpy as np
from torch import nn,optim
from torch.autograd import  Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
#训练集
train_dataset = datasets.MNIST(root=r'./',
                               train=True,
                               transform = transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root=r'./',
                               train=False,
                               transform = transforms.ToTensor(),
                               download=True)
#每次训练传入64个数据
batch_size = 64
#装载数据集
train_loader = DataLoader(dataset = train_dataset,
                          batch_size=batch_size,
                          shuffle=True)#打乱数据
test_loader = DataLoader(dataset = test_dataset,
                          batch_size=batch_size,
                          shuffle=True)#打乱数据
for i,data in enumerate(train_loader):
    inputs,labels = data
    print(inputs.shape)
    print(labels.shape)
    break
#定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(784,10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        #[64,1,28,28]->[64,784]
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x
LR = 0.5
#定义模型
model = Net()
#定义代价函数
mse_loss  = nn.CrossEntropyLoss()
#定义优化器
optimizer = optim.SGD(model.parameters(),LR)

def train():
    for i,data in enumerate(train_loader):
        #获得一个批次的数据和标签
        inputs, labels = data
        #获得模型预测结果（64，10）
        out = model(inputs)
        #变成独热编码
        # #(64)->(64,1)
        # labels = labes.reshape(-1,1)
        # #tensor.scatter_(dim, index, src)
        # #dim对哪个维度进独热编码
        # #src:插入index的数值
        # one_hot = torch.zeros(inputs.shape[0],10).scatter(1,labels,1)
        #计算loss,mes_loss的两个数据的shape要一致
        #交叉熵代价函数out(batch,C),labels(batch)
        loss = mse_loss(out,labels)
        #梯度清零
        optimizer.zero_grad()
        #计算梯度
        loss.backward()
        #修改权值
        optimizer.step()


def test():
    correct = 0
    for i,data in enumerate(test_loader):
        inputs,labels = data
        #获得模型预测结果
        out = model(inputs)
        #获得最大值以及其所在的位置
        _,predicted = torch.max(out,1)
        #预测正确的数量
        correct +=(predicted == labels).sum()
    print('Test acc:{0}'.format(correct.item()/len(test_dataset)))

for epoch in range(10):
    print('epoch',epoch)
    train()
    test()