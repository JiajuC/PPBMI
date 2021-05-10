import numpy as np
import os
from torch import nn,optim
import torch
from output_log import Logger
from draw_while_running import draw_while_running
from load_data import load_adult_dataset,construct_data_loader
from models.adult_models import EncoderModel,SimulationDecoderModel,SimulationTopModel,DecoderModel,TopModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
seed = 1
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
'''

class LocalModel(nn.Module):
    def __init__(self,encoder,decoder,top_model,train_loader,attribute_number):
        super(LocalModel, self).__init__()
        self.device = 'cuda'
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.top_model = top_model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 0.001
        self.optimizer_top = optim.Adam(top_model.parameters(), lr=self.lr)
        self.optimizer_encoder = optim.Adam(encoder.parameters(), lr=self.lr)
        self.optimizer_decoder = optim.Adam(decoder.parameters(), lr=self.lr)
        self.train_loader = train_loader
        self.attribute_number = attribute_number

    def train_model(self):
        total_epoch = 30
        train_loader = self.train_loader
        device = self.device
        lam = 0.95

        for epoch in range(total_epoch):
            print("train local model {}/{}".format(epoch+1, total_epoch))
            correct = 0
            for batch_idx, (inputs,targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                private_attributes = inputs[:,self.attribute_number]
                features = self.encoder(inputs)
                inference_attributes = self.decoder(features)
                outputs = self.top_model(features)
                acc_loss = self.criterion(outputs,targets.long())
                pri_loss = self.criterion(inference_attributes,private_attributes.long())

                #print("training simulative decoder")
                self.optimizer_decoder.zero_grad()
                pri_loss.backward(retain_graph=True)
                self.optimizer_decoder.step()

                #print("training simulative top model")
                self.optimizer_top.zero_grad()
                acc_loss.backward(retain_graph=True)
                self.optimizer_top.step()

                #print("training encoder")
                inference_attributes = self.decoder(features)
                outputs = self.top_model(features)
                new_acc_loss = self.criterion(outputs,targets.long())
                new_pri_loss = self.criterion(inference_attributes,private_attributes.long())
                loss = lam * new_acc_loss - (1-lam) * new_pri_loss
                self.optimizer_encoder.zero_grad()
                loss.backward()
                self.optimizer_encoder.step()

                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            train_accuracy = correct / float(len(train_loader.dataset))
            print("train accuracy = {:.2f}%".format(train_accuracy * 100))


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_adult_dataset()
    train_loader = construct_data_loader(train_data, train_label, 32)
    test_loader = construct_data_loader(test_data, test_label, 32)
    private_attribute_number = 1
    device = 'cuda'

    # train local model
    encoder = EncoderModel()
    simulation_decoder = SimulationDecoderModel(private_attribute_number)
    simulation_topmodel = SimulationTopModel()
    local_model = LocalModel(encoder,simulation_decoder,simulation_topmodel,train_loader,private_attribute_number)
    local_model.train_model()
    print("local model training over")

    #torch.save(encoder,'encoder1.pkl')

    # get feature after the encoder is fixed
    train_feature = encoder(train_data.to(device)).detach()
    test_feature = encoder(test_data.to(device)).detach()
    top_train_loader = construct_data_loader(train_feature, train_label, 32)
    top_test_loader = construct_data_loader(test_feature, test_label, 32)

    # train top model to test the utility of the encoded feature
    top_model = TopModel(top_train_loader, top_test_loader)
    top_model.train_model()
    top_model.test_model()

    # prepare dataset for the decoder
    train_private_label = train_data[:, private_attribute_number].to(device)
    test_private_label = test_data[:, private_attribute_number].to(device)
    decoder_test_loader = construct_data_loader(train_feature, train_private_label, 32)
    decoder_train_loader = construct_data_loader(test_feature, test_private_label, 32)

    # train decoder to test the privacy of the encoded feature
    decoder = DecoderModel(decoder_train_loader, decoder_test_loader,private_attribute_number)
    decoder.train_decoder()
    decoder.test_privacy()
