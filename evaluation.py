import torch
from models import CNN
from torch.utils.data import DataLoader
from torch import nn

target_model = CNN(10)
target_model.load_state_dict(torch.load('models/target_model.mod', weights_only=False))
target_model.eval()

# example
# output = target_model(input)