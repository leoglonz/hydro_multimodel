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

# Model_path =  "/projects/mhpi/yxs275/model/water_loss_model/" + '/dPL_local_daymet_new_attr_water_loss_v3/'
# out = os.path.join(Model_path, "exp_EPOCH50_BS100_RHO365_HS512_trainBuff365")
rootOut = "/projects/mhpi/yxs275/model/"+'LSTM_local_daymet_filled_withNaN_NSE_with_same_forcing_HBV_2800/'
out = os.path.join(rootOut, "exp_EPOCH300_BS100_RHO365_HS512_trainBuff365/")  # output folder to save results


NWM_gage_folder  = glob.glob('/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/daily_simulation_updated'+"/*")
NWM_gage_all =[x.split("/")[-1] for x in NWM_gage_folder]

NWM_gage_all = [str(int(x)) for x in NWM_gage_all]

zarr_save_path = '/projects/mhpi/yxs275/Data/GAGES_2/zarr/'



all_time_range = pd.date_range(start=f'{1980}-01-01', end=f'{2019}-12-31', freq='D')


with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_info_all_gages.json') as f:
    area_info_all_gages = json.load(f)

warmup = len(pd.date_range('1980-01-01',f'1982-12-31', freq='d'))
testing_timespan = pd.date_range('1980-01-01',f'2019-12-31', freq='d')

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


forcing_start_id = all_time_range.get_loc(testing_timespan[0])
forcing_end_id = all_time_range.get_loc(testing_timespan[-1])+1

trained_foring_selected = np.full((len(NWM_gage_all),len(testing_timespan),len(var_x_list)),np.nan)

trained_attr_selected = np.full((len(NWM_gage_all),len(attributeLst)),np.nan)
for gageidx, gage in enumerate(NWM_gage_all):
    gage = str(int(gage))
    gage_root = zarr.open_group(zarr_save_path+gage, mode = 'r')
    for forcingi, forcing_name in enumerate(var_x_list):   
        if  forcing_name == 'observation':
            trained_foring_selected[gageidx,:, forcingi] = gage_root[forcing_name]['discharge'][forcing_start_id:forcing_end_id]
        else:
            trained_foring_selected[gageidx,:, forcingi] = gage_root[forcing_name][forcing_start_id:forcing_end_id]     

    for attri, attr_name in enumerate(attributeLst): 
        if attr_name == 'area':
            trained_attr_selected[gageidx,attri] = attributeALL_df[attributeALL_df['STATID']==int(gage)]['DRAIN_SQKM'].values
        else:
             
            trained_attr_selected[gageidx,attri] = gage_root['attr'][attr_name][:]

streamflow = trained_foring_selected[:,:,-1:]
basin_area = trained_attr_selected[:,np.where(np.array(attributeLst)=='area')[0]]
mean_prep = trained_attr_selected[:,np.where(np.array(attributeLst)=='meanP')[0]]/365


forcing = trained_foring_selected[:,:,:-1]


streamflow_trans = scale._basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


attribute = trained_attr_selected
xTrain = scale.fill_Nan(forcing)
log_norm_cols=[]

## Input normalization
with open(out + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)
#stat_dict["catchsize"] = stat_dict["area"]
forcing_LSTM_norm = scale._trans_norm(
    xTrain.copy(), ['prcp','tmean','pet'], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)
forcing_LSTM_norm [forcing_LSTM_norm != forcing_LSTM_norm] = 0
forcing_LSTM_norm = np.concatenate([forcing_LSTM_norm[:,:warmup,:],forcing_LSTM_norm], axis = 1)

attribute_norm = scale._trans_norm(attribute, list(attributeLst), stat_dict, log_norm_cols=log_norm_cols, to_norm=True)
attribute_norm[attribute_norm != attribute_norm] = 0


# load the model

testepoch = 300
model_path = out
print("Load model from ", model_path)
testmodel = loadModel(model_path, epoch=testepoch)


testbatch =500 #len(indexes)

filePathLst = [out+"/NWM_Qs"]




testmodel.inittime = 0
train.testModel(
    testmodel, forcing_LSTM_norm, None, c=attribute_norm,  area_info = area_info_all_gages ,gage_key = NWM_gage_all,  batchSize=testbatch, filePathLst=filePathLst)


dataPred = pd.read_csv(  out+"/NWM_Qs", dtype=np.float32, header=None).values
dataPred = np.expand_dims(dataPred, axis=-1)

yPred = scale._trans_norm(
    dataPred[:, :, 0 :  1].copy(),
    ['usgsFlow'],
    stat_dict,
    log_norm_cols=log_norm_cols,
    to_norm=False,
)

yPred = scale._basin_norm_for_LSTM(
                        yPred.copy(), basin_area, mean_prep, to_norm=False
                    )


yPred_mmday = scale._basin_norm(
                        yPred.copy(), basin_area, to_norm=True
                    )




evaDict = [stat.statError(yPred_mmday[:,warmup:,0], streamflow_trans[:,:,0])]
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


print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))





data_arrays = {}

   
dataPred = np.expand_dims(dataPred, axis=-1)
data_array = xr.DataArray(
    yPred_mmday[:,warmup:,0],
    dims = ['COMID','time'],
    coords = {'COMID':NWM_gage_all,
                'time':testing_timespan}
)

data_arrays['LSTM_Qs'] = data_array


data_array = xr.DataArray(
    streamflow_trans[:,:,0],
    dims = ['COMID','time'],
    coords = {'COMID':NWM_gage_all,
                'time':testing_timespan}
)

data_arrays['runoff'] = data_array


data_array = xr.DataArray(
    streamflow[:,:,0],
    dims = ['COMID','time'],
    coords = {'COMID':NWM_gage_all,
                'time':testing_timespan}
)

data_arrays['streamflow'] = data_array

xr_dataset = xr.Dataset(data_arrays)
xr_dataset.to_zarr(store=out, group=f'NWM_simulation', mode='w')