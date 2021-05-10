import numpy as np
import os
from torch import nn,optim
import torch
from output_log import Logger
from draw_while_running import draw_while_running
from load_data import load_adult_dataset,construct_data_loader
from models.adult_models import SimulationDecoderModel,DecoderModel,TopModel


if __name__ == '__main__':
    encoder = torch.load('encoder.pkl')
    simulation_decoder = SimulationDecoderModel()