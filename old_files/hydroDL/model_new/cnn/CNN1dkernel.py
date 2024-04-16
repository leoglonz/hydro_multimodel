import csv
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model_new.dropout import DropMask, createMask
from torch.nn import Parameter


class CNN1dkernel(torch.nn.Module):
    def __init__(self, *, ninchannel=1, nkernel=3, kernelSize=3, stride=1, padding=0):
        super(CNN1dkernel, self).__init__()
        self.cnn1d = torch.nn.Conv1d(
            in_channels=ninchannel,
            out_channels=nkernel,
            kernel_size=kernelSize,
            padding=padding,
            stride=stride,
        )
        self.name = "CNN1dkernel"
        self.is_legacy = True

    def forward(self, x):
        output = F.relu(self.cnn1d(x))
        return output
