import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))

from hydroDL.model import rnn, crit, train
from hydroDL.data import scale,PET

from hydroDL.post import plot, stat
from hydroDL.master import loadModel
from sklearn.model_selection import KFold
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
import xarray as xr
import zarr
import time
import glob
import matplotlib.pyplot as plt
## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


## GPU setting
# which GPU to use when having multiple
traingpuid = 0
torch.cuda.set_device(traingpuid)

NWM_cal_basin_path = "/projects/mhpi/hjj5218/data/NWM/zarr/"


NWM_cal_basin_files  = glob.glob(NWM_cal_basin_path+"/*")
NWM_cal_basin_gage =[x.split("/")[-1] for x in NWM_cal_basin_files]
NWM_cal_basin_gage.sort()

# NWM_gage_folder  = glob.glob('/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/daily_simulation_updated'+"/*")
# NWM_gage_all =[x.split("/")[-1] for x in NWM_gage_folder]

# NWM_gage_all = [str(int(x)) for x in NWM_gage_all]

# NWM_gage = [x for x in NWM_gage_all if x not in NWM_cal_basin_gage]
# NWM_gage.sort()
# NWM_cal_basin_gage = NWM_gage

zarr_save_path = '/projects/mhpi/yxs275/Data/GAGES_2/zarr/'

forcing_zarr = '/projects/mhpi/hjj5218/data/NWM/basin_average.zarr/'
#forcing_zarr = "/projects/mhpi/hjj5218/data/NWM/zarr/"
all_time_range = pd.date_range(start=f'{1980}-01-01', end=f'{2019}-12-31', freq='D')

forcing_time_range = pd.date_range('1979-02-01',f'2023-01-31', freq='d')

# warmup = len(pd.date_range('1980-10-01',f'1982-09-30', freq='d'))
# testing_timespan = pd.date_range('1980-10-01',f'2000-09-30', freq='d')

warmup = len(pd.date_range('1988-10-01',f'2000-09-30', freq='d'))
testing_timespan = pd.date_range('1988-10-01',f'2019-09-30', freq='d')

Q_start_id = all_time_range.get_loc(testing_timespan[0])
Q_end_id = all_time_range.get_loc(testing_timespan[-1])+1

forcing_start_id = forcing_time_range.get_loc(testing_timespan[0])
forcing_end_id = forcing_time_range.get_loc(testing_timespan[-1])+1


# AROC_merit_zarr_path = '/projects/mhpi/data/NWM/ciroh-rti-public-data/merit_nwm_v30_retro/daily/'
# AROC_merit_root = zarr.open_group(AROC_merit_zarr_path+'0_1999', mode = 'r')
# AROC_COMID = [int(x) for x in AROC_merit_root['COMID']]

# daymet_zarr_path = '/projects/mhpi/data/conus/zarr/71/'
# daymet_merit_root = zarr.open_group(daymet_zarr_path+'71_0', mode = 'r')
# daymet_COMID = [int(x) for x in daymet_merit_root['COMID']]

# [C, ind1, SubInd] = np.intersect1d(AROC_COMID, daymet_COMID, return_indices=True)
# AROC_MERIT = AROC_merit_root['P'][ind1,forcing_start_id:forcing_end_id ]
# DAYMET_MERIT = daymet_merit_root['P'][SubInd,Q_start_id:Q_end_id]

# AROC_MERIT_df = pd.DataFrame(AROC_MERIT.transpose(), index=training_timespan)  ###from mm. s-1 to mm.h-1
# annual_sum_AROC_df = AROC_MERIT_df.resample('A-SEP').sum()

# annual_sum_AROC = np.nanmean(annual_sum_AROC_df.values.transpose(),axis = 1)

# DAYMET_MERIT_df = pd.DataFrame(DAYMET_MERIT.transpose(), index=training_timespan)  ###from mm. s-1 to mm.h-1
# annual_sum_DAYMET_df = DAYMET_MERIT_df.resample('A-SEP').sum()

# annual_sum_DAYMET = np.nanmean(annual_sum_DAYMET_df.values.transpose(),axis = 1)
# bias = (annual_sum_AROC -annual_sum_DAYMET )/annual_sum_DAYMET
# #aa = len(np.where(abs(annual_sum_AROC -annual_sum_DAYMET )/annual_sum_DAYMET <0.1)[0])
# finite_bias  = bias [np.isfinite(bias )]
# plt.figure(figsize=(6, 6))
# plt.hist(finite_bias, bins=30, edgecolor='black')
# plt.title('Histogram of Weekly NSE')
# plt.xlabel('Weekly NSE')
# plt.ylabel('Frequency')
# plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/'+ f"forcing_bias.png", dpi=300)



# def NSE_calc(pred,target):
#     ngrid,nt = pred.shape    
#     NSE = np.full(ngrid, np.nan)
#     for k in range(0, ngrid):
#         x = pred[k, :]
#         y = target[k, :]
#         ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
#         if ind.shape[0] > 0:
#             xx = x[ind]
#             yy = y[ind]
#             yymean = yy.mean()
#             SST = np.sum((yy-yymean)**2)
#             SSRes = np.sum((yy-xx)**2)
#             NSE[k] = 1-SSRes/SST

#     return NSE


# weekly_sum_DAYMET_df = DAYMET_MERIT_df.resample('W').sum()
# weekly_sum_AROC_df = AROC_MERIT_df.resample('w').sum()
# NSE = NSE_calc(weekly_sum_AROC_df .values.transpose(),weekly_sum_DAYMET_df .values.transpose())
# finite_NSE = NSE[np.isfinite(NSE)]

# # # Plot histogram
# plt.figure(figsize=(6, 6))
# plt.hist(finite_NSE, bins=30, edgecolor='black')
# plt.title('Histogram of Weekly NSE')
# plt.xlabel('Weekly NSE')
# plt.ylabel('Frequency')
# plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/'+ f"forcing_NSE.png", dpi=300)


# i = 1500
# plt.figure(figsize=(12, 6))
# plt.plot(training_timespan, AROC_MERIT[i, :], label='AROC', color='blue')
# plt.plot(training_timespan, DAYMET_MERIT[i, :], label='DAYMET', color='orange')
# plt.title(f'Time Series of {np.array(AROC_COMID)[i]}, weekly NSE {NSE[i]}')
# plt.xlabel('Date')
# plt.ylabel('Precipitation (mm)')
# plt.legend()
# plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/'+ f"forcing_Ts.png", dpi=300)




with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_info_all_gages.json') as f:
    area_info_all_gages = json.load(f)



var_x_list = ['P','Temp','PET','observation']

attributeLst = ['area','ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']


attribute_file = '/projects/mhpi/yxs275/Data/GAGES_2/all_gages_info.csv'
attributeALL_df = pd.read_csv(attribute_file)





trained_foring_selected = np.full((len(NWM_cal_basin_gage),len(testing_timespan),len(var_x_list)),np.nan)
trained_foring_selected_daymet = np.full((len(NWM_cal_basin_gage),len(testing_timespan),len(var_x_list)),np.nan)
trained_attr_selected = np.full((len(NWM_cal_basin_gage),len(attributeLst)),np.nan)
for gageidx, gage in enumerate(NWM_cal_basin_gage):
   
    AROC_root = zarr.open_group(forcing_zarr+gage.zfill(8), mode = 'r')

    gage = str(int(gage))
    gage_root = zarr.open_group(zarr_save_path+gage, mode = 'r')
    
    for forcingi, forcing_name in enumerate(var_x_list):   
        if  forcing_name == 'observation':
            trained_foring_selected[gageidx,:, forcingi] = gage_root[forcing_name]['discharge'][Q_start_id:Q_end_id]
            trained_foring_selected_daymet[gageidx,:, forcingi] = gage_root[forcing_name]['discharge'][Q_start_id:Q_end_id]
        else:
            trained_foring_selected[gageidx,:, forcingi] = AROC_root[forcing_name][forcing_start_id:forcing_end_id,0]     
            trained_foring_selected_daymet[gageidx,:, forcingi] = gage_root[forcing_name][Q_start_id:Q_end_id]

    for attri, attr_name in enumerate(attributeLst): 
        if attr_name == 'area':
            trained_attr_selected[gageidx,attri] = attributeALL_df[attributeALL_df['STATID']==int(gage)]['DRAIN_SQKM'].values
        else:
             
            trained_attr_selected[gageidx,attri] = gage_root['attr'][attr_name][:]

streamflow = trained_foring_selected[:,:,-1:]
basin_area = trained_attr_selected[:,np.where(np.array(attributeLst)=='area')[0]]
forcing_train = trained_foring_selected[:,:,:-1]


streamflow_trans = scale._basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


attr_train = trained_attr_selected

log_norm_cols=[]


rootOut = "/projects/mhpi/yxs275/model/NWM"+'/dHBV_NWM_calibration_basins_NSE_wo_log_with_cr_dy_K0_spatial_AROC_area_weighted/'

out = os.path.join(rootOut, "exp_EPOCH100_BS100_RHO365_HS256_trainBuff365/")  # output folder to save results

with open(out + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)

forcing_LSTM_norm = scale._trans_norm(
    forcing_train.copy(), var_x_list[:-1], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)

forcing_train[forcing_train!=forcing_train]  = 0

forcing_LSTM_norm[forcing_LSTM_norm!=forcing_LSTM_norm]  = 0

attribute_norm = scale.trans_norm(attr_train, list(attributeLst), stat_dict, to_norm=True)
attribute_norm[attribute_norm!=attribute_norm] = 0

attribute_norm = np.expand_dims(attribute_norm, axis=1)
attribute_norm = np.repeat(attribute_norm, forcing_LSTM_norm.shape[1], axis=1)




testepoch = 50
model_path = out
print("Load model from ", model_path)
testmodel = loadModel(model_path, epoch=testepoch)

zTest = np.concatenate([forcing_LSTM_norm, attribute_norm], 2)  # Add attributes to historical forcings as the inversion part
xTest = forcing_train
testTuple = (xTest, zTest)
testbatch =600 #len(indexes)


filePathLst = [out+"/Qs.csv",out+"/Q0.csv",out+"/Q1.csv",out+"/Q2.csv",out+"/ET.csv",out+"/SWE.csv"]



testmodel.inittime = 0

train.testModel(
    testmodel, testTuple, None, c=None, batchSize=testbatch, filePathLst=filePathLst)



dataPred = pd.read_csv(  out+"/Qs.csv", dtype=np.float32, header=None).values
dataPred = np.expand_dims(dataPred, axis=-1)


evaDict = [stat.statError(dataPred[:,warmup:,0], streamflow_trans[:,warmup:,0])]
evaDictLst = evaDict
keyLst = ['NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)


print("dHBV model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))
