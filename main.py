import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import os

class Data_in:

    def __init__(self):
        self.path = os.getcwd() + "/inputs/"


