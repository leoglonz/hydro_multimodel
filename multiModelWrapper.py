# This file contains a wrapper for the PRMS (Song, Rahmani) and HBV models 
# (MHPI Team). Eventually modified SAC-SMA will also be included (Song, Rahmani).
# Purpose: To evaluate the potential benefits of a multi-ensemble in "multi-model"
# and "multi-model" forms (see definitions below).
#
# 
# Last revised: 7 Jan. 2024
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F
from hydroDL.post import plot, stat

Ttest = [19951001, 20051001]
## Choose multi-model type: 'ensemble', 'mean', 'mosaic'. 
mType = 'ensemble'



path = []

# Path to HBV predictions.
path.append('D:/data/model_runs/rnnStreamflow/CAMELSDemo/dPLHBV/ALL/Testforc/daymet/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_12_Buff_365_Mul_16/pred50.npy')
# Path to PRMS predictions.
path.append('D:/data/model_runs/rnnStreamflow/CAMELSDemo/PRMS/pred50.npy')
# Path to SAC-SMA predictions.
path.append('D:/data/model_runs/rnnStreamflow/CAMELSDemo/SACSMA/pred50.npy')

# Calculate num days in testing period.
dateTest1 = datetime.strptime(str(Ttest[0]), '%Y%m%d')
dateTest2 = datetime.strptime(str(Ttest[1]), '%Y%m%d')
delta_test = dateTest2 - dateTest1
num_days_test = delta_test.days

if (mType=='ensemble'):
    # Ensemble := take the avg of model preds (streamflow) at each location.
    ""


elif (mType=='mean'):
    # Mean := take the avg of model preds (streamflow) at each location.

    # Initializing composite tensor of streamflow test predictions.
    predtestALL = np.full([671, num_days_test, 5, len(path)], np.nan)

    for i in range(len(path)):
        pred = np.load(path[i])
        predtestALL[:,:,:,i] = pred

    # Fetching observed streamflow values.
    obstestALL = np.load(
        'D:/data/model_runs/rnnStreamflow/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para1.0/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/obs.npy')

    # Average all the ensemble predictions.
    # The avg of axis 3 of form [i,j,k,m] represents the avg for the [i, j, k]-th 
    # attribute across all models m in the ensemble. (dim reduction)
    predtestALL_allEn = np.mean(predtestALL, axis=3)
    evaDict = [stat.statError(predtestALL_allEn[:, :, 0], obstestALL.squeeze())]  # Q0: the streamflow
    evaframe = pd.DataFrame(evaDict[0])

    print('For all basins, NSE median across models:', np.nanmedian(evaDict[0]['NSE']))

    ## Show boxplots of the results.
    evaDictLst = evaDict
    plt.rcParams['font.size'] = 14
    plt.rcParams["legend.columnspacing"] = 0.1
    plt.rcParams["legend.handletextpad"] = 0.2
    keyLst = ['NSE', 'KGE', 'lowRMSE', 'highRMSE']
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(evaDictLst)):
            data = evaDictLst[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)

    # Return model stats.
    print("NSE,KGE,lowRMSE, highRMSE, mean lowRMSE, and highRMSE of all basins: ", np.nanmedian(dataBox[0][0]),
        np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
        np.nanmean(dataBox[2][0]), np.nanmean(dataBox[3][0]))

    labelname = ['dPL+HBV_Multi', 'dPL+HBV_Multi Sub531']
    xlabel = keyLst
    fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(6, 5))
    fig.patch.set_facecolor('white')
    fig.show()
    # plt.savefig(os.path.join(outpath, 'Metric_BoxPlot.png'), format='png')


elif (mType=="mosaic"):
    # Mosaic := take the softmax of model preds (streamflow) at each location.

    def m_softmax(z, axis=1):
        assert len(z.shape) == 4  # Ensure 4D Tensor.
        s = np.max(z, axis=axis)
        s = np.expand_dims(s, axis=axis)  # Necessary step for broadcasting.
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=axis)
        div = np.expand_dims(div, axis=axis)
        return e_x / div

else:
    raise(ValueError("Invalid model type."))