import numpy as np
import os
from torch import nn,optim
import torch
from output_log import Logger
from draw_while_running import draw_while_running
from load_data import load_adult_dataset,construct_data_loader
from models.adult_models import EncoderModel,SimulationDecoderModel,SimulationTopModel,DecoderModel,TopModel

class LocalModel(nn.Module):



if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_adult_dataset()
    train_loader, test_loader = construct_data_loader(train_data, train_label, test_data, test_label, 32)
    private_attribute_number = 1
    device = 'cuda'

    encoder = torch.load('encoder.pkl')
    simulation_decoder = SimulationDecoderModel(private_attribute_number)
