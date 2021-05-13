#https://github.com/dongyp13/Adversarial-Distributional-Training

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 76
step_size = 0.3
epsilon = 8.0 / 255.0
step_size = 0.3  # 学习率
num_steps = 7 # number of optimization steps
log_interval = 100  # 每100个batch log一次
lbd = 0.01  # lambda
seed = 1
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def adt_loss(model, x_natural, y, optimizer, learning_rate=1.0, epsilon=8.0 / 255.0, perturb_steps=10, num_samples=10,
             lbd=0.0):
    model.eval()  # 外层模型先不动，先优化内层
    mean = Variable(torch.zeros(x_natural.size()).cuda(), requires_grad=True)  # 平均值 miu
    var = Variable(torch.zeros(x_natural.size()).cuda(), requires_grad=True)  # 方差
    optimizer_adv = torch.optim.Adam([mean, var], lr=learning_rate, betas=(0.0, 0.0))  # 内部optimizer

    for _ in range(perturb_steps):
        for s in range(num_samples):  # 优化（5）max部分
            adv_std = F.softplus(var)  # 标准差 sigma
            rand_noise = torch.randn_like(x_natural)  # r
            adv = torch.tanh(mean + rand_noise * adv_std)

            negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
            # 忽略常数项后的（10）式第二项

            entropy = negative_logp.mean()  # entropy
            x_adv = torch.clamp(x_natural + epsilon * adv, 0.0, 1.0)

            # minimize the negative loss
            with torch.enable_grad():
                loss = -F.cross_entropy(model(x_adv), y) - lbd * entropy
            loss.backward(retain_graph=True if s != num_samples - 1 else False)

        optimizer_adv.step()  # 更新参数

    x_adv = torch.clamp(x_natural + epsilon * torch.tanh(mean + F.softplus(var) * torch.randn_like(x_natural)), 0.0,
                        1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # 梯度清零
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    return loss


def train(model, device, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        loss = adt_loss(model=model,
                        x_natural=data,
                        y=target,
                        optimizer=optimizer,  # 外部采用SGD优化器
                        learning_rate=step_size,  # 学习率0.3
                        epsilon=epsilon,
                        perturb_steps=num_steps,  # 内部优化次数
                        num_samples=5,  # MC采样次数
                        lbd=lbd)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

if __name__ == '__main__':
    momentum = 0.9  # SGD下降冲量
    lr = 0.1  # SGD学习率
    weight_decay = 2e-4  # SGD权重下降率
    train_loader, test_loader =
    model =
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

        #eval_test
