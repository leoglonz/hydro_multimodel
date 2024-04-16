import csv
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model_new import rnn
from hydroDL.model_new.dropout import DropMask, createMask
from torch.nn import Parameter


class CudnnInvLstmModel(torch.nn.Module):
    # using cudnnLstm to extract features from SMAP observations
    def __init__(self, *, nx, ny, hiddenSize, ninv, nfea, hiddeninv, dr=0.5, drinv=0.5):
        # two LSTM
        super(CudnnInvLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.lstminv = rnn.CudnnLstmModel(
            nx=ninv, ny=nfea, hiddenSize=hiddeninv, dr=drinv
        )
        self.lstm = rnn.CudnnLstmModel(
            nx=nfea + nx, ny=ny, hiddenSize=hiddenSize, dr=dr
        )
        self.gpu = 1
        self.name = "CudnnInvLstmModel"
        self.is_legacy = True

    def forward(self, x, z, doDropMC=False):
        Gen = self.lstminv(z)
        dim = x.shape
        nt = dim[0]
        invpara = Gen[-1, :, :].repeat(nt, 1, 1)
        x1 = torch.cat((x, invpara), dim=2)
        out = self.lstm(x1)
        # out = rho/time * batchsize * Ntargetvar
        return out
