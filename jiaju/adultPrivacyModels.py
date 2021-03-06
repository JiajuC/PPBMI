import numpy as np
from torch import nn,optim
import torch
import os
from tqdm import tqdm
from lossFunction import privacyLoss1,privacyLoss2,privacyLoss3
import matplotlib.pyplot as plt
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = 'cuda'
SIGMA = 1

class Net(nn.Module):
    def __init__(self,encoder,topModel,train_loader,test_loader,type,featureDim,lam2):
        super(Net,self).__init__()
        self.device = 'cuda'
        self.encoder = encoder.to(self.device)
        self.topModel = topModel.to(self.device)
        self.criterrion = torch.nn.CrossEntropyLoss()
        if type == 2:
            self.privacyLoss = privacyLoss2(sigma=1,device= device,lam=lam2)
        elif type==1:
            self.privacyLoss = privacyLoss1(sigma=1,device = device)
        else:
            self.privacyLoss = privacyLoss3(sigma=1,device = device)
        self.lr = 0.001

        self.cov = torch.eye(featureDim)*0.5*SIGMA#Covariance matrix which is initialized to one
        self.optimizerTop = optim.Adam(topModel.parameters(),lr=self.lr)
        self.optimizerEncoder = optim.Adam(encoder.parameters(),lr = self.lr)
        self.trainLoader = train_loader
        self.testLoader = test_loader
        self.type = type

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

    def trainModel(self,lam = 0.1,pan=0,lam2=0.25):
        train_loader = self.trainLoader
        device = self.device
        trainLoss = 0
        correct = 0
        total = 0

        for batch_idx,(inputs,targets) in enumerate(train_loader):

            inputs,targets = inputs.to(device),targets.to(device)

            feature = self.getOutputFeature(inputs)#torch.float32,torch.Size([32, 16])
            if self.type != 0:
                feature = self.getGaussian(feature)
            topOutputs = self.topModel(feature)
            accLoss = self.criterrion(topOutputs,targets.long())


            if self.type == 3:
                pL = self.privacyLoss(feature, targets,pan)
                loss = (accLoss + lam*pL.to(device)).to(device)
            elif self.type == 2 or self.type == 1:
                privateAttributeLabel = inputs[:, pan]
                pL = self.privacyLoss(feature, privateAttributeLabel, pan)
                loss = (accLoss + lam * pL.to(device)).to(device)
                # print("pri:",lam *pL)
                # print("acc:",accLoss)
            else:
                loss = accLoss
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
            if self.type != 0:
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
    def __init__(self,attNum,featureDim):
        super(EncoderModel,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(attNum,32),
            # nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(32, featureDim),
            # nn.Dropout(p=0.5),
            nn.ReLU(True)
        )

    def forward(self,x):
        x = x.view(x.size()[0],-1)
        x = self.layer1(x)
        return x

class TestTopModel(nn.Module):
    def __init__(self,featureDim,taskNum):
        super(TestTopModel,self).__init__()
        self.top_layer = nn.Sequential(
            nn.Linear(featureDim,16),
            nn.ReLU(True),
            nn.Linear(16, taskNum)
        )

    def forward(self,x):
        output = self.top_layer(x)
        return output

class DecoderModel(nn.Module):
    def __init__(self,train_loader,test_loader,attribute_number,type_num,featureDim):
        super(DecoderModel, self).__init__()
        self.type_num = type_num
        self.model = nn.Sequential(
            nn.Linear(featureDim,32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16,self.type_num[attribute_number])
        )
        self.trainLoader = train_loader
        self.testLoader = test_loader
        self.device = 'cuda'
        self.lr = 0.001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def train_decoder(self,decoderEpoch = 30):
        epoch_num = decoderEpoch
        device = self.device
        train_loss_list = []
        pdb = tqdm(range(epoch_num))
        for epoch in pdb:
            pdb.set_description("train decoder {}/{}".format(epoch, epoch_num))
            # train decoder
            decoder_train_loss = 0.0
            decoder_train_correct = 0
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(self.trainLoader):
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)

                outputs = self.model(decoder_inputs)
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, decoder_targets.long())
                train_loss_list.append(loss.item())
                # loss_list.append(float(loss))
                loss.backward()
                self.optimizer.step()
                _, predicted = outputs.max(1)
                decoder_train_correct += predicted.eq(decoder_targets).sum().item()
                # calculate accuracy
                decoder_train_loss += loss.item() * decoder_inputs.size(0)
            train_loss = decoder_train_loss / float(len(self.trainLoader.dataset))
            # print('train privacy', train_loss_list)
        plt.plot(train_loss_list)
        plt.savefig(r"decoder.png")
        # plt.show()

    def test_decoder(self):
        device = self.device
        decoder_test_correct = 0
        with torch.no_grad():
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(self.testLoader):
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(
                    device)
                outputs = self.model(decoder_inputs)
                _, predicted = outputs.max(1)
                decoder_test_correct += predicted.eq(decoder_targets).sum().item()
        test_privacy = 1-decoder_test_correct / float(len(self.testLoader.dataset))
        print('test privacy', test_privacy)
        return test_privacy

class TopModel(nn.Module):
    def __init__(self,train_loader,test_loader,featureDim,taskNum):
        super(TopModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(featureDim, 16),
            nn.ReLU(True),
            nn.Linear(16,taskNum),
            #nn.Softmax(dim=1)
        )
        self.device = 'cuda'
        self.lr = 0.001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader


    def train_model(self,topModelEpoch=30):
        total_epoch = topModelEpoch
        train_loader = self.train_loader
        device = self.device
        pdb = tqdm(range(total_epoch))

        for epoch in pdb:
            pdb.set_description("train top model {}/{}".format(epoch + 1, total_epoch))
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
            pdb.set_postfix(train_accuracy = train_acc * 100)


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




