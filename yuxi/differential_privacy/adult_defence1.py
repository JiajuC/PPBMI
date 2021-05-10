import numpy as np
import os
from torch import nn,optim
import torch
from output_log import Logger
from draw_while_running import draw_while_running
from load_data import load_adult_dataset,construct_data_loader
from models.adult_models import SimulationDecoderModel,DecoderModel,TopModel


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def add_gaussian_noise(inputs, mean, std):
    noise = inputs.data.new(inputs.size()).normal_(mean, std)
    return inputs + noise


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_adult_dataset()
    train_loader = construct_data_loader(train_data, train_label, 32)
    test_loader = construct_data_loader(test_data, test_label, 32)
    private_attribute_number = 1
    noise_variance = 100
    device = 'cuda'

    encoder = torch.load('encoder.pkl')
    simulation_decoder = SimulationDecoderModel(private_attribute_number)

    # get feature after the encoder is fixed
    train_feature = encoder(train_data.to(device)).detach()
    test_feature = encoder(test_data.to(device)).detach()

    # add gaussian noise to encoded feature
    noisy_train_feature = add_gaussian_noise(train_feature, 0, noise_variance)
    noisy_test_feature = add_gaussian_noise(test_feature, 0, noise_variance)

    top_train_loader = construct_data_loader(noisy_train_feature, train_label, 32)
    top_test_loader = construct_data_loader(noisy_test_feature, test_label, 32)

    # train top model to test the utility of the encoded feature
    top_model = TopModel(top_train_loader, top_test_loader)
    top_model.train_model()
    top_model.test_model()

    # prepare dataset for the decoder
    test_private_label = train_data[:, private_attribute_number].to(device)
    train_private_label = test_data[:, private_attribute_number].to(device)
    decoder_train_feature = noisy_test_feature
    decoder_test_feature = noisy_train_feature
    decoder_train_loader = construct_data_loader(decoder_train_feature, train_private_label, 32)
    decoder_test_loader = construct_data_loader(decoder_test_feature, test_private_label, 32)

    # train decoder to test the privacy of the encoded feature
    decoder = DecoderModel(decoder_train_loader, decoder_test_loader, private_attribute_number)
    decoder.train_decoder()
    decoder.test_privacy()