import numpy as np
import os
from torch import nn,optim
import torch
from load_data import get_data_loader


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 0
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class AdultModel(nn.Module):
    def __init__(self,train_loader,test_loader):
        super(AdultModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 32),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(32, 64),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            #nn.Softmax(dim=1)
        )
        self.device = 'cuda'
        self.lr = 0.001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self):
        train_loader = self.train_loader
        device = self.device

        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets.long())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = correct / float(len(train_loader.dataset))
        print("train accuracy = {:.2f}%".format(train_accuracy * 100))

    def test_model(self):
        test_loader = self.test_loader
        device = self.device
        correct = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

        test_accuracy = correct / float(len(test_loader.dataset))
        print("test accuracy = {:.2f}%".format(test_accuracy * 100))


if __name__ == '__main__':
    train_loader, test_loader = get_data_loader('adult')
    adult_model = AdultModel(train_loader,test_loader)

    total_epoch = 20
    for epoch in range(total_epoch):
        print("train model {}/{}".format(epoch + 1, total_epoch))

        adult_model.train_model()
        adult_model.test_model()
