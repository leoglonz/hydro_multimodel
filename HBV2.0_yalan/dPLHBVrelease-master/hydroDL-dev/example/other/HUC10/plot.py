
import sys
sys.path.append('../../')
import hydroDL
from hydroDL.post import plot
import numpy as np
import time
import json
import os

from scipy.interpolate import griddata
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

import string
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import pandas as pd
import json


data_folder = "/data/yxs275/CONUS_data/HUC10/dPL_1_11_2024/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/"
# parameter_folder = "/data/yxs275/DPL_HBV/HUC10_Output/dPL/"
attribute_folder = "/data/yxs275/CONUS_data/HUC10/version_1_11_2014_continental_routing/"
attributeALL_df = pd.read_csv(attribute_folder + "attributes.csv")
basinID = attributeALL_df.gage_ID.values
batchSize = 1000
iS = np.arange(0, len(basinID), batchSize)
iE = np.append(iS[1:], len(basinID))

date_all = pd.date_range('1980-01-01',f'2021-01-01', freq='d', closed='left')

# for item in range(len(iS)):
#     file = parameter_folder + f"para_{iS[item]}_{iE[item]}.npy"
#     para = np.load(file)
#     if item ==0:
#         paraAll = para
#     else:
#         paraAll = np.concatenate((paraAll,para),axis = 0)
for item in range(len(iS)):
    file = data_folder + f"Percolation_{iS[item]}_{iE[item]}"
    dataPred = pd.read_csv(file, dtype=np.float32, header=None)
    dataPred_Qr =  dataPred.values

    attributeBatch_file = attribute_folder + f"attributes_{iS[item]}_{iE[item]}.csv"
    attributeBatch_df = pd.read_csv(attributeBatch_file)


    # file = data_folder + f"Q2_{iS[item]}_{iE[item]}"
    # dataPred = pd.read_csv(file, dtype=np.float32, header=None)
    # dataPred_Q2 =  dataPred.values

    for year in range(1981,2021):
        date_year = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='d')
        startIdx = date_all.get_loc(f'{year}-01-01')
        endIdx = date_all.get_loc(f'{year}-12-31')
        annual_value_Qr = np.expand_dims(dataPred_Qr[:, startIdx:endIdx+1].sum(-1), axis=-1)
        #annual_value_Q2 = np.expand_dims(dataPred_Q2[:, startIdx:endIdx].sum(-1), axis=-1)
        #annual_value = annual_value_Q2/annual_value_Qr
        if year == 1981:

            dataPred_year = annual_value_Qr
        else:
            dataPred_year = np.concatenate((dataPred_year,annual_value_Qr),axis = -1 )
    #forcingBatch = np.load(data_folder + f"forcings_{iS[item]}_{iE[item]}.npy")
    #dataPred_year = dataPred_Q2.sum(-1)/ dataPred_Qr.sum(-1)
    dataPred_year = dataPred_year.mean(-1)
    if item ==0:
        QrAll = dataPred_year
        latAll = attributeBatch_df.lat.values
        lonAll = attributeBatch_df.lon.values
    else:
        QrAll = np.concatenate((QrAll,dataPred_year),axis = 0)
        latAll = np.concatenate((latAll, attributeBatch_df.lat.values), axis=0)
        lonAll = np.concatenate((lonAll, attributeBatch_df.lon.values), axis=0)


fontsize =14
plt.rcParams.update({'font.size': fontsize})
figsize = [8, 5]
fig = plt.figure(figsize=figsize)
# gs = gridspec.GridSpec(3+nAx, nMap)
gs = gridspec.GridSpec(1, 1)
mapColor = None
nMap = 1
site_list =[]
for k in range(nMap):

    # ax = fig.add_subplot(gs[0:2, k])
    nrow = 0
    ncol = 0
    crsProj = ccrs.PlateCarree()  # set geographic cooridnates
    ax = fig.add_subplot(gs[nrow, ncol], projection=crsProj, frameon=True)

    cRange = None
    title = "Annual recharge (mm/year)"
    data = QrAll

    # title = r"$FC (mm)$"
    # scale = [50,1000]
    # data = scale[0]+paraAll[:,1]*(scale[1]-scale[0])
    textBool = False
    if k == 0: textBool = True
    #textLst = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]
    textLst = ["(i)", "(ii)", "(iii)"]
    plot.plotMap(data,  lat=latAll, lon=lonAll, ax=ax, cRange=cRange, title=title)
plt.savefig("/data/yxs275/CONUS_data/HUC10/dPL_1_11_2024/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/" + "Annual_recharge.png", dpi=300)
plt.show(block=True)