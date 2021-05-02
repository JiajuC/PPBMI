import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import time
from output_log import Logger
from draw_while_running import draw_while_running
import pickle
import copy

FEATURE_NUMBER = 0
BATCH_SIZE = 32
POISON_RATIO = 0.05

seed = 1
device = 'cuda'
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# data preprocessing
def data_preprocess():
    data_path = '../../Data/adult/'
    f1 = open(data_path + 'train_data.pickle', 'rb')
    f2 = open(data_path + 'train_label.pickle', 'rb')
    f3 = open(data_path + 'test_data.pickle', 'rb')
    f4 = open(data_path + 'test_label.pickle', 'rb')
    X_train = np.array(pickle.load(f1), dtype='float32')
    y_train = pickle.load(f2)
    X_test = np.array(pickle.load(f3), dtype='float32')
    y_test = pickle.load(f4)
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    train_data = torch.from_numpy(X_train)
    party_A_train_data = train_data[:, :5]
    party_B_train_data = train_data[:, 5:]
    train_label = torch.from_numpy(y_train)

    test_data = torch.from_numpy(X_test)
    party_A_test_data = test_data[:, :5]
    party_B_test_data = test_data[:, 5:]
    test_label = torch.from_numpy(y_test)

    poison_ratio = POISON_RATIO
    poison_mark = torch.zeros_like(train_label)
    poison_mark[:int(poison_ratio * train_label.shape[0])] = 1

    train_dataset = TensorDataset(party_A_train_data, party_B_train_data, train_label, poison_mark)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(party_A_test_data, party_B_test_data, test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

class Vertical_FL_Model(nn.Module):
    def __init__(self, bottomA, bottomB, top_model, train_loader, test_loader):
        super(Vertical_FL_Model, self).__init__()
        self.device = 'cuda'
        self.bottomA = bottomA.to(self.device)
        self.bottomB = bottomB.to(self.device)
        self.top_model = top_model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.decoder_criterion = torch.nn.MSELoss()
        self.lr = 0.0001
        self.optimizer_top = optim.Adam(top_model.parameters(), lr=self.lr)
        self.optimizerA = optim.Adam(bottomA.parameters(), lr=self.lr)
        self.optimizerB = optim.Adam(bottomB.parameters(), lr=self.lr)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.feature_number = FEATURE_NUMBER

    def calculate_output_feature(self, A_inputs):
        feature = self.bottomA(A_inputs)
        return feature

    def train_model(self):
        train_loader = self.train_loader
        device = self.device
        feature_number = self.feature_number
        train_loss = 0
        correct = 0
        total = 0
        decoder_train_data = torch.tensor([]).to(device)
        decoder_train_label = torch.tensor([]).to(device)
        decoder_test_data = torch.tensor([]).to(device)
        decoder_test_label = torch.tensor([]).to(device)
        for batch_idx, (A_inputs, B_inputs, targets, poison_mark) in enumerate(train_loader):
            A_inputs, B_inputs, targets = A_inputs.to(device), B_inputs.to(device), targets.to(device)
            featureA = self.calculate_output_feature(A_inputs)
            featureB = self.bottomB(B_inputs)
            poison_index = torch.nonzero(poison_mark).flatten().to(device)

            decoder_train_data_part = torch.index_select(featureA.detach(), 0, poison_index)
            decoder_train_label_part = torch.index_select(A_inputs.detach(), 0, poison_index)
            decoder_test_data_part = torch.tensor(
                np.delete(np.array(featureA.detach().to('cpu')), np.array(poison_index.to('cpu')), 0)).to(device)
            decoder_test_label_part = torch.tensor(
                np.delete(np.array(A_inputs.detach().to('cpu')), np.array(poison_index.to('cpu')), 0)).to(device)

            decoder_train_data = torch.cat([decoder_train_data, decoder_train_data_part.detach()], dim=0)
            decoder_train_label = torch.cat([decoder_train_label, decoder_train_label_part.detach()], dim=0)
            decoder_test_data = torch.cat([decoder_test_data, decoder_test_data_part.detach()], dim=0)
            decoder_test_label = torch.cat([decoder_test_label, decoder_test_label_part.detach()], dim=0)

            top_outputs = self.top_model(featureA, featureB)
            acc_loss = self.criterion(top_outputs, targets.long())
            self.optimizer_top.zero_grad()
            self.optimizerA.zero_grad()
            self.optimizerB.zero_grad()
            acc_loss.backward()
            self.optimizer_top.step()
            self.optimizerA.step()
            self.optimizerB.step()

            # calculate accuracy
            train_loss += acc_loss.item()
            _, predicted = top_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_accuracy = correct / float(len(train_loader.dataset))
        print("train accuracy = {:.2f}%".format(train_accuracy * 100))

        # print(decoder_train_data.shape)
        # print(decoder_test_data.shape)

        # construct decoder dataset
        decoder_train_label = decoder_train_label[:, feature_number]
        decoder_test_label = decoder_test_label[:, feature_number]
        decoder_train_dataset = TensorDataset(decoder_train_data, decoder_train_label)
        decoder_train_loader = DataLoader(dataset=decoder_train_dataset, batch_size=128, shuffle=True)
        decoder_test_dataset = TensorDataset(decoder_test_data, decoder_test_label)
        decoder_test_loader = DataLoader(dataset=decoder_test_dataset, batch_size=128, shuffle=False)

        return train_accuracy, decoder_train_loader, decoder_test_loader

    def test_model(self):
        test_loader = self.test_loader
        device = self.device
        test_loss = 0
        correct = 0
        total = 0
        # with torch.no_grad():
        for batch_idx, (A_inputs, B_inputs, targets) in enumerate(test_loader):
            A_inputs, B_inputs, targets = A_inputs.to(device), B_inputs.to(device), targets.to(device)
            featureA = self.calculate_output_feature(A_inputs)
            featureB = self.bottomB(B_inputs)
            top_outputs = self.top_model(featureA, featureB)
            loss = self.criterion(top_outputs, targets.long())

            test_loss += loss.item()
            _, predicted = top_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_accuracy = correct / float(total)
        print("test accuracy = {:.2f}%".format(test_accuracy * 100))

        return test_accuracy


class BottomModelA(nn.Module):
    def __init__(self):
        super(BottomModelA, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True)
        )

    def forward(self, A_input):
        partyA_output = self.layers(A_input)
        return partyA_output


class BottomModelB(nn.Module):
    def __init__(self):
        super(BottomModelB, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True)
        )

    def forward(self, B_input):
        partyB_output = self.layers(B_input)
        return partyB_output


class TopModel(nn.Module):
    def __init__(self):
        super(TopModel, self).__init__()
        self.interactive_layerA = nn.Linear(64, 32)
        self.interactive_layerB = nn.Linear(64, 32)
        self.interactive_activation_layer = nn.ReLU(True)
        self.top_layer = nn.Sequential(nn.Linear(64, 2))

    def forward(self, A_output, B_output):
        interactive_activation_input = torch.cat((self.interactive_layerA(A_output), self.interactive_layerB(B_output)),
                                                 1)
        interactive_output = self.interactive_activation_layer(interactive_activation_input)
        output = self.top_layer(interactive_output)
        return output


class OutsideDecoder(nn.Module):
    def __init__(self):
        super(OutsideDecoder, self).__init__()
        self.type_num = [9,4,9,7,15]
        self.feature_number = FEATURE_NUMBER
        self.model = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(True),
            nn.Linear(32,self.type_num[self.feature_number])
        )
        self.device = 'cuda'
        self.lr = 0.0001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def train_decoder(self, decoder_train_loader):
        epoch_num = 100
        device = self.device
        for epoch in range(epoch_num):
            print("train decoder {}/{}".format(epoch, epoch_num))
            # train decoder
            decoder_train_loss = 0.0
            decoder_train_correct = 0
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(decoder_train_loader):
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)
                outputs = self.model(decoder_inputs)
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, decoder_targets.long())
                # loss_list.append(float(loss))
                loss.backward()
                self.optimizer.step()
                _, predicted = outputs.max(1)
                decoder_train_correct += predicted.eq(decoder_targets).sum().item()
                # calculate accuracy
                decoder_train_loss += loss.item() * decoder_inputs.size(0)
            train_privacy = (1 - (decoder_train_correct / float(len(decoder_train_loader.dataset))))
            print('train privacy', train_privacy)
        return train_privacy

    def test_decoder(self,decoder_test_loader):
        device = self.device
        decoder_test_correct = 0
        with torch.no_grad():
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(decoder_test_loader):
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(
                    device)
                outputs = self.model(decoder_inputs)
                _, predicted = outputs.max(1)
                decoder_test_correct += predicted.eq(decoder_targets).sum().item()
        test_privacy = (1-(decoder_test_correct / float(len(decoder_test_loader.dataset))))
        print('test privacy', test_privacy)
        return test_privacy



# if __name__ == 'main':
file_name = 'adult'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
outputSavePath = './test' + file_name + '_' + timestamp
if not os.path.exists(outputSavePath):
    os.mkdir(outputSavePath)
logSavePath = outputSavePath + '/log'
if not os.path.exists(logSavePath):
    os.mkdir(logSavePath)
sys.stdout = Logger(os.path.join(logSavePath, "output.txt"), sys.stdout)
sys.stderr = Logger(os.path.join(logSavePath, "error.txt"), sys.stderr)
rewardSavePath = outputSavePath + '/saveReward'
if not os.path.exists(rewardSavePath):
    os.mkdir(rewardSavePath)
results_name = 'results_log.txt'
accuracy_file = open(os.path.join(rewardSavePath, results_name), 'w')

train_loader, test_loader = data_preprocess()
partyA = BottomModelA()
partyB = BottomModelB()
top_model = TopModel()
outside_decoder_initial = OutsideDecoder()
#outside_decoder = OutsideDecoder()

vertical_model = Vertical_FL_Model(partyA, partyB, top_model, train_loader, test_loader)
total_epoch = 30
for epoch in range(total_epoch):
    print("Epoch {}".format(epoch))
    outside_decoder = copy.deepcopy(outside_decoder_initial)
    train_accuracy, decoder_train_loader, decoder_test_loader = vertical_model.train_model()
    test_accuracy = vertical_model.test_model()
    train_privacy = outside_decoder.train_decoder(decoder_train_loader)
    test_privacy = outside_decoder.test_decoder(decoder_test_loader)

    # write into file
    # save log
    result_file = open(os.path.join(rewardSavePath, results_name), 'a')

    result_file.write(
        str(epoch) + ' ' + str(train_privacy) + ' ' + str(test_privacy) + ' ' + str(train_accuracy) + ' '
        + str(test_accuracy) + ' ' + '\n')
    result_file.close()

    #
    # # draw
    if epoch > 0:
        draw_while_running(rewardSavePath, results_name, rewardSavePath, str(epoch) + '_results.svg',
                           'train_vertical_model',
                           'epoch',
                           'results',
                           ['epoch', 'train_privacy', 'test_privacy', 'train_accuracy', 'test_accuracy'])
