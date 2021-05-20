import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from load_data import load_adult_dataset, construct_data_loader
from adult_models import EncoderModel, SimulationDecoderModel, SimulationTopModel, DecoderModel, TopModel
import matplotlib.pyplot as plt
from torch import nn, optim


lbd = 0.2


def train_simulation_decoder(decoder, inputs, targets, epsilon):
    mean = Variable(torch.zeros(inputs.size()).cuda(), requires_grad=True)
    var = Variable(torch.zeros(inputs.size()).cuda(), requires_grad=True)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=0.01)
    optimizer_adv = optim.Adam([mean, var], lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    loss_list = []
    perturb_steps = 10
    num_samples = 10
    total_epoch = 20

    for epoch in range(total_epoch):
        print("adversarial training {}/{}".format(epoch + 1, total_epoch))
        for _ in range(perturb_steps):
            for s in range(num_samples):
                adv_std = F.softplus(var)
                rand_noise = torch.rand_like(inputs)
                adv = torch.tanh(mean + rand_noise * adv_std)
                negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
                entropy = negative_logp.mean()
                x_adv = torch.clamp(inputs + epsilon * adv, 0.0, 1.0)

                pred1 = decoder(x_adv)
                loss1 = -criterion(pred1, targets.long()) - lbd * entropy
                loss_list.append(loss1.item())

                optimizer_decoder.zero_grad()
                loss1.backward(retain_graph=True)
                pred2 = decoder(x_adv)
                loss2 = -criterion(pred2, targets.long()) - lbd * entropy
                optimizer_adv.zero_grad()
                loss2.backward(retain_graph=True)
        optimizer_decoder.step()
        optimizer_adv.step()
        x_adv = torch.clamp(inputs + epsilon * torch.tanh(mean + F.softplus(var) * torch.randn_like(inputs)), 0.0, 1.0)

    plt.plot(loss_list)
    plt.show()


def add_perturbation_distribution(decoder, inputs, targets, epsilon):
    mean = Variable(torch.zeros(inputs.size()).cuda(), requires_grad=True)
    var = Variable(torch.zeros(inputs.size()).cuda(), requires_grad=True)
    optimizer_adv = optim.Adam([mean, var], lr=0.01, betas=(0.0, 0.0))
    loss_list = []
    total_epoch = 20
    perturb_steps = 10
    num_samples = 10
    for epoch in range(total_epoch):
        for _ in range(perturb_steps):
            for s in range(num_samples):
                adv_std = F.softplus(var)
                rand_noise = torch.rand_like(inputs)
                adv = torch.tanh(mean + rand_noise * adv_std)
                negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
                entropy = negative_logp.mean()
                x_adv = torch.clamp(inputs + epsilon * adv, 0.0, 1.0)

                with torch.enable_grad():
                    loss = -F.cross_entropy(decoder(x_adv), targets.long()) - lbd * entropy
                loss_list.append(loss.item())
                loss.backward(retain_graph=True)

        optimizer_adv.step()

    x_adv = torch.clamp(inputs + epsilon * torch.tanh(mean + F.softplus(var) * torch.randn_like(inputs)), 0.0, 1.0)

    plt.plot(loss_list)
    plt.show()

    return x_adv


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

    train_simulation_decoder(simulation_decoder, train_feature, train_private_label, epsilon)

    # get perturbed feature after the simulation decoder is fixed
    perturbed_train_feature = add_perturbation_distribution(simulation_decoder, train_feature, train_private_label,
                                                            epsilon)
    perturbed_test_feature = add_perturbation_distribution(simulation_decoder, test_feature, test_private_label,
                                                           epsilon)

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
