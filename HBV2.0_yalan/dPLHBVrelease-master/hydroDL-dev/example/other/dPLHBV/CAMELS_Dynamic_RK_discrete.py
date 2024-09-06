import sys
sys.path.append('../../')

from hydroDL.model import crit, train
from hydroDL.model import rnn_hourly as rnn
from hydroDL.data import camels
from hydroDL.post import plot, stat
import torch.nn.functional as F
import os
import numpy as np
import torch
from collections import OrderedDict
import random
import json
import pandas as pd
import json
import datetime as dt

## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

traingpuid = 4
torch.cuda.set_device(traingpuid)

data = np.load("/data/yxs275/NROdeSolver/CAMELSData/Camels_inputs.npy")
streamflow_data = np.load("/data/yxs275/NROdeSolver/CAMELSData/Camels_outputs.npy")

forcing_norm = np.load("/data/yxs275/NROdeSolver/CAMELSData/Camels_inputs_norm.npy")

attri_norm = np.load("/data/yxs275/NROdeSolver/CAMELSData/Camels_attr_norm.npy")


Ninv = forcing_norm.shape[-1]+attri_norm.shape[-1]
EPOCH = 50 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 10
alpha = 0.25
HIDDENSIZE = 256
BUFFTIME = 365 # for each training sample, to use BUFFTIME days to warm up the states.
routing = True # Whether to use the routing module for simulated runoff
Nmul = 16 # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
comprout = False # True is doing routing for each component
compwts = False # True is using weighted average for components; False is the simple mean
pcorr = None # or a list to give the range of precip correc

tdRep = [1, 13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
tdRepS = [str(ix) for ix in tdRep]
# ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
# Must set below ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
# If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
ETMod = True
Nfea = 13  # should be 13 when setting ETMod=True. 12 when ETMod=False
dydrop = 0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
staind = -1  # which time step to use from the learned para time series for those static parameters

model = rnn.MultiInv_HBVTDModel(ninv=Ninv, nfea=Nfea, nmul=Nmul, hiddeninv=HIDDENSIZE, inittime=BUFFTIME,
                                routOpt=routing, comprout=comprout, scheme = "discrete",hourstep = 24.0,compwts=compwts, staind=staind, tdlst=tdRep,
                                dydrop=dydrop, ETMod=ETMod)
lossFun = crit.RmseLossComb(alpha=alpha)

forcTuple = [data,forcing_norm]

rootOut = "/data/yxs275/DPL_HBV/output/"+'/record_16_RK_discrete/'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, f"exp_EPOCH{EPOCH}_BS{BATCH_SIZE}_RHO{RHO}_HS{HIDDENSIZE}_trainBuff{BUFFTIME}") # output folder to save results
if os.path.exists(out) is False:
    os.mkdir(out)

trainedModel = train.trainModel(
    model,
    forcTuple,
    streamflow_data,
    attri_norm,
    lossFun,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=BUFFTIME)