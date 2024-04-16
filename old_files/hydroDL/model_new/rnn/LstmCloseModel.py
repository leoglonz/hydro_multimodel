import csv
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model_new import rnn
from hydroDL.model_new.dropout import DropMask, createMask
from torch.nn import Parameter


class LstmCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, fillObs=True):
        super(LstmCloseModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx + 1, hiddenSize)
        # self.lstm = CudnnLstm(
        #     inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm = rnn.LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod="drW"
        )
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.fillObs = fillObs
        self.name = "LstmCloseModel"
        self.is_legacy = True

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out
