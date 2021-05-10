# there are 5 models in total:
# 1.an encoder (the main model we want to train)
# 2.a SIMULATIVE decoder (used to perform adversarial training with the encoder to decrease the mutual information)
# 3.a SIMULATIVE top model (used to simulate the downstream tasks to ensure the accuracy)
# 4.a REAL decoder (used to perform the real attacks from the outside attackers to test the privacy of the model)
# 5.a REAL top model (used to perform the real downstream tasks to test the accuracy of the model)
# the whole LOCAL model is consists of model 1~3 , model 4 is an unknown attacker and model 5 is a cloud server

from torch import nn,optim
import torch
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 32),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(32, 16),
            # nn.Dropout(p=0.5),
            nn.ReLU(True)
        )

    def forward(self,x):
        feature = self.layers(x)
        return feature


class SimulationDecoderModel(nn.Module):
    def __init__(self,attribute_number):
        super(SimulationDecoderModel, self).__init__()
        self.type_num = [9,4,9,7,15]
        self.layers = nn.Sequential(
            nn.Linear(16, 32),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(32, 16),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(16, self.type_num[attribute_number]),
            #nn.Softmax(dim=1)
        )

    def forward(self,feature):
        inference_attribute = self.layers(feature)
        return inference_attribute


class SimulationTopModel(nn.Module):
    def __init__(self):
        super(SimulationTopModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8,2),
            #nn.Softmax(dim=1)
        )

    def forward(self,feature):
        output = self.layers(feature)
        return output


class DecoderModel(nn.Module):
    def __init__(self,train_loader,test_loader,attribute_number):
        super(DecoderModel, self).__init__()
        self.type_num = [9, 4, 9, 7, 15]
        self.model = nn.Sequential(
            nn.Linear(16, 32),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(32, 16),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(16, self.type_num[attribute_number]),
            #nn.Softmax(dim=1)
        )
        self.device = 'cuda'
        self.lr = 0.001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_decoder(self):
        total_epoch = 30
        train_loader = self.train_loader
        device = self.device
        train_loss_list = []

        for epoch in range(total_epoch):
            print("train decoder {}/{}".format(epoch+1, total_epoch))
            decoder_train_loss = 0.0
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(train_loader):
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)
                outputs = self.model(decoder_inputs)
                loss = self.criterion(outputs, decoder_targets.long())
                train_loss_list.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                decoder_train_loss += loss.item() * decoder_inputs.size(0)
            train_loss = decoder_train_loss / float(len(train_loader.dataset))

        plt.plot(train_loss_list)
        plt.show()

    def test_privacy(self):
        test_loader = self.test_loader
        device = self.device
        correct = 0
        with torch.no_grad():
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(test_loader):
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)
                outputs = self.model(decoder_inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(decoder_targets).sum().item()

        privacy = 1 - correct/float(len(test_loader.dataset))
        print("test privacy = ",privacy )
        #return privacy


class TopModel(nn.Module):
    def __init__(self,train_loader,test_loader):
        super(TopModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8,2),
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
        total_epoch = 20
        train_loader = self.train_loader
        device = self.device

        for epoch in range(total_epoch):
            print("train top model {}/{}".format(epoch + 1, total_epoch))
            train_loss = 0.0
            correct = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()*inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            train_acc = correct / float(len(train_loader.dataset))
            print("train accuracy = {:.2f}%".format(train_acc * 100))

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

        test_acc = correct/float(len(test_loader.dataset))
        print("test accuracy = {:.2f}%".format(test_acc * 100))