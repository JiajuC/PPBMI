import numpy as np
import os
from torch import nn,optim
import torch
from output_log import Logger
from draw_while_running import draw_while_running
from load_data import load_adult_dataset,construct_data_loader
from models.adult_models import SimulationDecoderModel,SimulationTopModel,DecoderModel,TopModel
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

def train_simulation_decoder(decoder,inputs,targets,epsilon):
    device = 'cuda'
    delta = torch.zeros_like(inputs,requires_grad=True).to(device)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=0.01)
    optimizer_delta = optim.Adam([delta], lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    loss_list = []

    total_epoch = 2000
    for epoch in range(total_epoch):
        print("adversarial training {}/{}".format(epoch + 1, total_epoch))

        pred1 = decoder(inputs + delta)
        loss1 = criterion(pred1,targets.long())
        loss_list.append(loss1.item())

        optimizer_decoder.zero_grad()
        loss1.backward()
        optimizer_decoder.step()

        pred2 = decoder(inputs + delta)
        loss2 = -criterion(pred2,targets.long())

        optimizer_delta.zero_grad()
        loss2.backward()
        optimizer_delta.step()

        delta.data.clamp_(-epsilon, epsilon)

    plt.plot(loss_list)
    plt.show()


def add_perturbation(decoder,inputs,targets,epsilon):
    device = 'cuda'
    delta = torch.zeros_like(inputs,requires_grad=True).to(device)
    optimizer_delta = optim.Adam([delta], lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    loss_list = []

    total_epoch = 3000
    for epoch in range(total_epoch):
        pred = decoder(inputs + delta)
        loss = -criterion(pred, targets.long())
        loss_list.append(loss.item())

        optimizer_delta.zero_grad()
        loss.backward()
        optimizer_delta.step()
        delta.data.clamp_(-epsilon, epsilon)

    plt.plot(loss_list)
    plt.show()

    return inputs + delta.detach()


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_adult_dataset()
    private_attribute_number = 1
    device = 'cuda'
    epsilon = 2

    encoder = torch.load('encoder.pkl')
    simulation_decoder = SimulationDecoderModel(private_attribute_number).to(device)

    # get feature after the encoder is fixed
    train_feature = encoder(train_data.to(device)).detach()
    test_feature = encoder(test_data.to(device)).detach()

    train_private_label = train_data[:, private_attribute_number].to(device)
    test_private_label = test_data[:, private_attribute_number].to(device)

    train_simulation_decoder(simulation_decoder,train_feature,train_private_label,epsilon)

    # get perturbed feature after the simulation decoder is fixed
    perturbed_train_feature = add_perturbation(simulation_decoder,train_feature,train_private_label,epsilon)
    perturbed_test_feature = add_perturbation(simulation_decoder,test_feature,test_private_label,epsilon)

    # train top model to test the utility of the perturbed feature
    top_train_loader = construct_data_loader(perturbed_train_feature, train_label, 32)
    top_test_loader = construct_data_loader(perturbed_test_feature, test_label, 32)
    top_model = TopModel(top_train_loader, top_test_loader)
    top_model.train_model()
    top_model.test_model()


    # train decoder to test the privacy of the perturbed feature
    decoder_train_loader = construct_data_loader(perturbed_test_feature, test_private_label, 32)
    decoder_test_loader = construct_data_loader(perturbed_train_feature, train_private_label, 32)

    correct = 0
    with torch.no_grad():
        for batch_idx, (decoder_inputs, decoder_targets) in enumerate(decoder_test_loader):
            decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)
            outputs = simulation_decoder(decoder_inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(decoder_targets).sum().item()

    privacy = 1 - correct / float(len(decoder_test_loader.dataset))
    print("test privacy = ", privacy)


    decoder = DecoderModel(decoder_train_loader, decoder_test_loader, private_attribute_number)
    decoder.train_decoder()
    decoder.test_privacy()





