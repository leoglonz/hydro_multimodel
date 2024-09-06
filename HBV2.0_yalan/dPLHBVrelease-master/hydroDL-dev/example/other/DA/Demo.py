# This code is written by Yalan Song from MHPI group, Penn State University
# Purpose: Data assimilation

import sys
sys.path.append('../../')

import numpy as np
import torch
import random
import torch.nn as nn
import hydroDL
from hydroDL.model import rnn, cnn, crit
from hydroDL.post import plot, stat
traingpuid = 7
torch.cuda.set_device(traingpuid)

randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###Save parameters :30 years
path_model = '/data/yxs275/NROdeSolver/output/HBVtest_module_hbv_1_13_dynamic_rout_static/'
paraHBVFile = path_model + "discrete_static_paraHBV.npy"
paraRoutFile = path_model + "discrete_static_paraRout.npy"
allBasinParaHBV =  np.load(paraHBVFile)
allBasinParaRout = np.load(paraRoutFile)
###Save forcing for HBV function and LSTM inputs for LSTM :30 years
HBV_forcing = np.load("/data/yxs275/NROdeSolver/CAMELSData/HBV_test.npy")

LSTM_input_norm = np.load("/data/yxs275/NROdeSolver/CAMELSData/LSTM_test.npy" )

## Save streamflow simulation w/o DA  : (later 15 years in the all 30 years)
path_result_discrete = "/data/yxs275/DPL_HBV/output/record_16/CAMELSDemo/dPLHBV/ALL/" \
                       "TDTestforc/TD1_13/daymet/BuffOpt0/RMSE_para0.25/111111/" \
                       "Train19801001_19951001Test19951001_20101001Buff5478Staind5477/"

resultFile = path_result_discrete + "pred50.npy"
streamflow_prediction = np.load(resultFile)

## Save streamflow observation : (later 15 years in the all 30 years)
streamflow = np.load("/data/yxs275/NROdeSolver/CAMELSData/obs_test.npy")

##Hyperparameters
Ninv = LSTM_input_norm.shape[-1]
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

tdRep = [1,13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
tdRepS = [str(ix) for ix in tdRep]
# ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
# Must set belGR4Jow ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
# If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
ETMod = True
Nfea = 11  # should be 13 when setting ETMod=True. 12 when ETMod=False
dydrop = 0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
staind = -1  # which time step to use from the learned para time series for those static parameters

window_size = 10
window_size_lower = 30
batch_size= 100
num_epochs = 200
DA_days = 1100
## Step1: Warmup the storages with the first 15 years data
HBV_warmup = rnn.HBVMulET()

xinit = HBV_forcing[:batch_size,:-streamflow_prediction.shape[1],:,]
xinit = torch.from_numpy( np.swapaxes(xinit, 1, 0)).float()
xinit = xinit.cuda()
buffpara = allBasinParaHBV[-streamflow_prediction.shape[1],:batch_size,:,:]
buffpara = torch.from_numpy( buffpara).float()
buffpara = buffpara.cuda()
rtwts = allBasinParaRout[:batch_size,:]
rtwts= torch.from_numpy( rtwts).float()
rtwts = rtwts.cuda()
Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = HBV_warmup(xinit, buffpara, Nmul, None, rtwts, bufftime=0, outstate=True, routOpt=False, comprout=False)


## Step2: DA
x_DA = HBV_forcing[:batch_size,-streamflow_prediction.shape[1]:,:,]
x_DA =  np.swapaxes(x_DA, 1, 0)
x_DA_adjusted = x_DA.copy()

y_DA = streamflow[:batch_size,:,0:1]
y_DA =  np.swapaxes(y_DA, 1, 0)

yp_DA = streamflow_prediction[:batch_size,:,0:1]
yp_DA =  np.swapaxes(yp_DA, 1, 0)

yp_DA_adjusted = yp_DA.copy()

para_HBV = allBasinParaHBV[-streamflow_prediction.shape[1]:,:batch_size,:,:]

parstaFull = para_HBV[staind, :, :, :][np.newaxis].repeat(streamflow_prediction.shape[1], axis=0)
parhbvFull = parstaFull.copy()
for ix in tdRep:
    parhbvFull[:, :, ix - 1, :] = para_HBV[:, :, ix-1, :]

SNOWPACK = SNOWPACK.detach()
MELTWATER = MELTWATER.detach()
SM = SM.detach()
SUZ = SUZ.detach()
SLZ = SLZ.detach()

dtype=torch.float64
saler_k = [0.8,1.2]
HBV = rnn.HBVMulTDET_DA()

lossFun = crit.RmseLossComb(alpha=0)

for i in range(window_size_lower,window_size_lower+DA_days):
    k  = nn.Parameter(torch.rand(batch_size).cuda())
    optimizer = torch.optim.Adadelta([k],lr = 20)
    optimizer.zero_grad()

    loss = 0
    for iepoch in range(num_epochs):
        k0 = torch.sigmoid(k)
        newk = saler_k[0] + k0 * (saler_k[1] - saler_k[0])

        window_forcing_p = torch.from_numpy(x_DA_adjusted[i-window_size_lower:i + window_size, :, :]).float().cuda()

        window_para_HBV = torch.from_numpy(parhbvFull[i-window_size_lower:i+window_size,:,:,:]).float().cuda()
        window_obs = torch.from_numpy(y_DA[i-window_size_lower:i+window_size,:,:]).float().cuda()
        window_ypretrain = torch.from_numpy(yp_DA[i-window_size_lower:i + window_size, :,  0:1]).float().cuda()

        window_forcing = window_forcing_p.clone()
        window_forcing[window_size_lower,:,0] = newk*window_forcing_p[window_size_lower,:,0]
        out = HBV( window_forcing, window_para_HBV, [SNOWPACK.detach(), MELTWATER.detach(), SM.detach(), SUZ.detach(), SLZ.detach()],staind, tdRep, Nmul, None, rtwts, bufftime=0, outstate=False, routOpt=True,
                comprout=False, dydrop=False)
        out_old = HBV( window_forcing_p, window_para_HBV, [SNOWPACK.detach(), MELTWATER.detach(), SM.detach(), SUZ.detach(), SLZ.detach()],staind, tdRep, Nmul, None, rtwts, bufftime=0, outstate=False, routOpt=True,
                comprout=False, dydrop=False)
        window_yp_DA = out[:,:,0:1]
        window_yp_old = out_old[:, :, 0:1]
        if iepoch % 5 == 0:
            loss_check = loss

        loss = lossFun(window_yp_DA[window_size_lower:,:,0:1], window_obs[window_size_lower:,:,0:1])
        loss_old = lossFun(window_yp_old[window_size_lower:, :, 0:1], window_obs[window_size_lower:, :, 0:1])
        loss_pretrain = lossFun(window_ypretrain[window_size_lower:,:,0:1], window_obs[window_size_lower:,:,0:1])


        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if (abs(loss-loss_check)<0.0001):
            break
        print('Window {}, iepoch {}: Loss {:.3f}'.format(i,iepoch, loss.item()))
    if (loss<loss_pretrain):

        Qsinit,SNOWPACK, MELTWATER, SM, SUZ, SLZ = HBV(window_forcing[0:1,:,:], window_para_HBV[0:1,:,:,:], [SNOWPACK.detach(), MELTWATER.detach(), SM.detach(), SUZ.detach(), SLZ.detach()],
                                                staind, tdRep, Nmul, None, rtwts,
                                                bufftime=0, outstate=True, routOpt=False,
                                                comprout=False, dydrop=False)

        x_DA_adjusted[i:i+1 , :, :] = window_forcing[window_size_lower:window_size_lower+1,:,:].detach().cpu().numpy()
        index = np.where(window_forcing[window_size_lower, :, 0].detach().cpu().numpy()<0.000001)

        yp_DA_adjusted[i:i + 1, :, :] = window_yp_DA[window_size_lower:window_size_lower+1,:,:].detach().cpu().numpy()
        yp_DA_adjusted[i:i + 1, index, :] = yp_DA[i:i + 1, index, :]
    else:

        Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = HBV(window_forcing_p[0:1,:,:], window_para_HBV[0:1,:,:,:],
                                                        [SNOWPACK.detach(), MELTWATER.detach(), SM.detach(), SUZ.detach(), SLZ.detach()],
                                                        staind, tdRep, Nmul, None, rtwts,
                                                        bufftime=0, outstate=True, routOpt=False,
                                                        comprout=False, dydrop=False)


evaDict = [stat.statError(np.swapaxes(yp_DA_adjusted[window_size_lower:window_size_lower+DA_days,:,0], 1, 0), np.swapaxes(y_DA[window_size_lower:window_size_lower+DA_days,:,0], 1, 0))]
keyLst = ['NSE', 'KGE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDict)):
        data = evaDict[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
print("NSE after adjustment is ",np.nanmedian( dataBox[0][0]) )

evaDict = [stat.statError(np.swapaxes(yp_DA[window_size_lower:window_size_lower+DA_days,:,0], 1, 0), np.swapaxes(y_DA[window_size_lower:window_size_lower+DA_days,:,0], 1, 0))]
keyLst = ['NSE', 'KGE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDict)):
        data = evaDict[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
print("NSE before adjustment is ",np.nanmedian( dataBox[0][0]) )
print("Done")