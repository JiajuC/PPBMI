import numpy as np
from torch import nn,optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import time
import pickle
import sys
import os
from lossFunction import privacyLoss1,privacyLoss2
from output_log import Logger
from draw_while_running import draw_while_running
seed = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda'
batchSize = 32
features = 16
cov = torch.eye(features)#Covariance matrix which is initialized to one


np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def dataPreprocess():
    dataPath = r'Data/adult/'
    f1 = open(dataPath + 'train_data.pickle', 'rb')
    f2 = open(dataPath + 'train_label.pickle', 'rb')
    f3 = open(dataPath + 'test_data.pickle', 'rb')
    f4 = open(dataPath + 'test_label.pickle', 'rb')
    X_train = np.array(pickle.load(f1), dtype='float32')
    y_train = pickle.load(f2)
    X_test = np.array(pickle.load(f3), dtype='float32')
    y_test = pickle.load(f4)
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    train_data = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train)
    test_data = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test)

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(test_data,test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.shape)
        print(labels.shape)
        break
    return train_loader, test_loader

class Net(nn.Module):
    def __init__(self,encoder,topModel,train_loader,test_loader):
        super(Net,self).__init__()
        self.device = 'cuda'
        self.encoder = encoder.to(self.device)
        self.topModel = topModel.to(self.device)
        self.criterrion = torch.nn.CrossEntropyLoss()
        self.privacyLoss = privacyLoss2(sigma=1,device= device)
        self.lr = 0.001
        self.batchSize = batchSize
        self.cov = cov
        self.optimizerTop = optim.Adam(topModel.parameters(),lr=self.lr)
        self.optimizerEncoder = optim.Adam(encoder.parameters(),lr = self.lr)
        self.trainLoader = train_loader
        self.testLoader = test_loader

    def getOutputFeature(self,inputs):
        feature = self.encoder(inputs)
        return feature

    def getGaussian(self, feature):
        t1 = feature.shape
        fx = t1[0]
        fy = t1[1]
        # feature1 = feature.cpu().detach().numpy()
        temp = np.random.multivariate_normal([0 for i in range(fy)], self.cov)
        # Data are generated according to covariance matrix and mean
        for i in range(1,fx):
            temp = np.concatenate((temp, np.random.multivariate_normal([0 for i in range(fy)],
                                                                       self.cov)),
                                  axis=0)
            # Splicing sampling of high dimensional Gaussian distribution data
        temp.resize((fx, fy))
        temp = torch.from_numpy(temp).float()
        # Since the stitched data is one-dimensional,
        # we redefine it as the original dimension
        feature = feature + temp.to(self.device)
        return feature

    def trainModel(self, mode=True):
        train_loader = self.trainLoader
        device = self.device
        trainLoss = 0
        correct = 0
        total = 0
        lam = 0.1
        for batch_idx,(inputs,targets) in enumerate(train_loader):
            inputs,targets = inputs.to(device),targets.to(device)
            feature = self.getOutputFeature(inputs)#torch.float32,torch.Size([32, 16])
            feature = self.getGaussian(feature)
            pL = self.privacyLoss(feature,targets)
            topOutputs = self.topModel(feature)
            accLoss = self.criterrion(topOutputs,targets.long())
            loss = (accLoss + lam*pL.to(device)).to(device)
            self.optimizerTop.zero_grad()
            self.optimizerEncoder.zero_grad()
            loss.backward()

            self.optimizerTop.step()
            self.optimizerEncoder.step()

            trainLoss +=loss.item()
            _,predicted = topOutputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        trainAcc = correct/float(len(train_loader.dataset))
        print("train accuracy={:.2f}%".format(trainAcc*100))
        return trainAcc

    def testModel(self):
        testLoader = self.testLoader
        device = self.device
        test_loss = 0
        correct = 0
        total = 0
        for batchIdx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            feature = self.getOutputFeature(inputs)
            feature = self.getGaussian(feature)
            topOutputs = self.topModel(feature)
            loss = self.criterrion(topOutputs, targets.long())

            test_loss += loss.item()
            _, predicted = topOutputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        testAcc = correct / float(total)
        print("test accuracy = {:.2f}%".format(testAcc * 100))

        return testAcc

class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(11,32),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(32, features),
            # nn.Dropout(p=0.5),
            nn.ReLU(True)
        )

    def forward(self,x):
        x = x.view(x.size()[0],-1)
        x = self.layer1(x)
        return x

class TopModel(nn.Module):
    def __init__(self):
        super(TopModel,self).__init__()
        self.top_layer = nn.Sequential(
            nn.Linear(features,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        output = self.top_layer(x)
        return output

if __name__ == '__main__':
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

    trainLoader, testLoader = dataPreprocess()
    encoder = EncoderModel()
    topModel = TopModel()
    model = Net(encoder,topModel,trainLoader,testLoader)
    totalEpoch = 50

    for epoch in range(totalEpoch):
        print("Epoch {}".format(epoch))
        trainAcc = model.trainModel()
        testAcc = model.testModel()
        result_file = open(os.path.join(rewardSavePath, results_name), 'a')

        result_file.write(
            str(epoch)  + '  ' + str(trainAcc) + '  '+ str(testAcc) + '  ' + '\n')
        result_file.close()

        #
        # # draw
        if epoch > 0:
            draw_while_running(rewardSavePath, results_name, rewardSavePath, str(epoch) + '_results.svg',
                               'train_vertical_model',
                               'epoch',
                               'results',
                               ['epoch', 'train_accuracy', 'test_accuracy'])
