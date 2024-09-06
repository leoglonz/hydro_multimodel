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

# Model_path =  "/projects/mhpi/yxs275/model/water_loss_model/" + '/dPL_local_daymet_new_attr_water_loss_v3/'
# out = os.path.join(Model_path, "exp_EPOCH50_BS100_RHO365_HS512_trainBuff365")
rootOut = "/projects/mhpi/yxs275/model/"+'LSTM_local_daymet_filled_withNaN_NSE_with_same_forcing_HBV_2800/'
out = os.path.join(rootOut, "exp_EPOCH300_BS100_RHO365_HS512_trainBuff365/")  # output folder to save results

with open(out + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)


log_norm_cols = []

# log_norm_cols=['prcp', 'pr', 'total_precipitation', 'pre', 'potential_evaporation', 'LE',
#                    'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'ET_water', 'ET_sum', 'susm', 'smp', 'ssma', 'susma',
#                    'usgsFlow', 'streamflow', 'qobs']

attributeLst_water_loss = ['ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']

forcing_list_water_loss = ['P','Temp','PET']


with open('/projects/mhpi/hjj5218/data/Main_River/CONUS_v3/'+'area_info_main_river.json') as f:
    area_info = json.load(f)

large_river_file = glob.glob("/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/large_river_daily_simulation/v2/"+"*")

River_id =[(x.split("/")[-1]) for x in large_river_file]
River_id.sort()

# bad_rivers =[
#     "02427500","02429500",'02470500', "03159870", "03160000", 
#     "06879000", "06888345", "07138062", "07138065", 
#     "07355500","08092600", "08447300", "09514300", "09158500",'09479501', 
#     '09514300','09518500',"09520280", "09520700", "12391000","13171620","13290200"
# ]


# for item in bad_rivers:
#     area_info.pop(item)

# attribut_river_file = "/projects/mhpi/hjj5218/data/Main_River/CONUS_v2/main_river_41.csv"
# attribut_river = pd.read_csv(attribut_river_file)
# area_river = attribut_river['area'].values

# obs_ft = np.load('/projects/mhpi/hjj5218/data/Main_River/CONUS_v2/obs_streamflow_1980_2020.npy')/0.0283168 

# obs_ft = np.swapaxes(obs_ft,1,0)
# obs_mm_day = scale._basin_norm(
#                         np.expand_dims(obs_ft,axis = -1 ) ,  np.expand_dims(area_river,axis = -1), to_norm=True
#                     )  ## from ft^3/s to mm/day




zarr_save_path = '/projects/mhpi/hjj5218/data/Main_River/CONUS_v2/distributed.zarr'

dHBV_time_range_all = pd.date_range(start=f'{1980}-01-01', end=f'{2020}-12-31', freq='D')

warmup_span = pd.date_range('1980-10-01',f'1981-09-30', freq='d')
test_span = pd.date_range('1981-10-01',f'2020-09-30', freq='d')

forcing_start_id = dHBV_time_range_all.get_loc(test_span[0])
forcing_end_id = dHBV_time_range_all.get_loc(test_span[-1])+1

obs_test = np.full((len(River_id),len(test_span),1),np.nan)
forcing_lumped = np.full((len(River_id),len(test_span),len(forcing_list_water_loss)),np.nan)

attr_lumped = np.full((len(River_id),len(attributeLst_water_loss)+1),np.nan)

# for gageidx, gagei in enumerate(area_info):

#     print(gagei, "has merit of ", len(area_info[gagei]['unitarea']))
#     print(gagei, "area is ", area_river[gageidx], "maximum area is ", np.max(area_info[gagei]['uparea']))

obs_path = '/projects/mhpi/hjj5218/data/Main_River/CONUS_v3/distributed.zarr/'

basin_area_all = np.full((len(River_id),1),np.nan)
for gageidx, gagei in enumerate(River_id):
    
    obs_gage_root = zarr.open_group(zarr_save_path+'/'+gagei, mode = 'r')
    COMIDs = obs_gage_root['COMID'][:]

    print(gagei, "has merit of ", len(COMIDs))

    unitarea = area_info[gagei]['unitarea']
    area_all = np.sum(unitarea)
    unitarea_fraction = unitarea/area_all
    unitarea_fraction_reshaped = unitarea_fraction[:, np.newaxis]
    root_obs = zarr.open_group(obs_path+gagei, mode = 'r')
    obs_i = root_obs['streamflow'][forcing_start_id:forcing_end_id]/0.0283168 
    basin_area = root_obs['DRAIN_SQKM'][:]

    basin_area_all[gageidx] = basin_area
    obs_mm_day = scale._basin_norm(
                        np.expand_dims(np.expand_dims(obs_i,axis = -1 ),axis = 0 ),  np.expand_dims(basin_area,axis = -1), to_norm=True
                    )  ## from ft^3/s to mm/day
    
    
    obs_test[gageidx,:,0] = obs_mm_day[0,:,0]



    

    xTrain2 = np.full((len(COMIDs),len(test_span),len(forcing_list_water_loss)),np.nan)
    attr2 = np.full((len(COMIDs),len(attributeLst_water_loss)),np.nan)


    for fid, foring_ in enumerate(forcing_list_water_loss):    
        xTrain2[:,:,fid] = obs_gage_root[foring_][:,forcing_start_id:forcing_end_id]
    # load attributes

    for aid, attribute_ in enumerate(attributeLst_water_loss) :                                    
        attr2[:,aid] =  obs_gage_root['attr'][attribute_][:] 

    forcing_lumped[gageidx,:,:] = np.nanmean(xTrain2, axis = 0)
    attr_lumped[gageidx,1:] = np.nanmean(attr2, axis = 0)
    attr_lumped[gageidx,0] = basin_area



forcing_norm = scale._trans_norm(
forcing_lumped,  ['prcp','tmean','pet'], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)

forcing_norm[forcing_norm!=forcing_norm]  = 0

attributeLst_water_loss_1 = ['area'] +attributeLst_water_loss

attr_norm = scale.trans_norm(attr_lumped, list(attributeLst_water_loss_1), stat_dict, to_norm=True)
attr_norm[attr_norm!=attr_norm] = 0


mean_prep = attr_lumped[:,np.where(np.array(attributeLst_water_loss_1)=='meanP')[0]]/365


# load the model

testepoch = 300
model_path = out
print("Load model from ", model_path)
testmodel = loadModel(model_path, epoch=testepoch)


testbatch =500 #len(indexes)

filePathLst = [out+"/large_river_Qs.csv"]




testmodel.inittime = 0
train.testModel(
    testmodel, forcing_norm, None, c=attr_norm,  area_info = area_info ,gage_key = list(area_info.keys()),  batchSize=testbatch, filePathLst=filePathLst)


dataPred = pd.read_csv(  out+"/large_river_Qs.csv", dtype=np.float32, header=None).values
dataPred = np.expand_dims(dataPred, axis=-1)



yPred = scale._trans_norm(
    dataPred[:, :, 0 :  1].copy(),
    ['usgsFlow'],
    stat_dict,
    log_norm_cols=log_norm_cols,
    to_norm=False,
)

yPred = scale._basin_norm_for_LSTM(
                        yPred.copy(), basin_area_all, mean_prep, to_norm=False
                    )


yPred_mmday = scale._basin_norm(
                        yPred.copy(), basin_area_all, to_norm=True
                    )




evaDict = [stat.statError(yPred_mmday[:,len(warmup_span):,0], obs_test[:,len(warmup_span):,0])]
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



test_span = pd.date_range('1981-10-01',f'2020-09-30', freq='d')

data_arrays = {}

   
dataPred = np.expand_dims(dataPred, axis=-1)
data_array = xr.DataArray(
    yPred_mmday[:,:,0],
    dims = ['COMID','time'],
    coords = {'COMID':River_id,
                'time':test_span}
)

data_arrays['LSTM_Qs'] = data_array


data_array = xr.DataArray(
    obs_test[:,:,0],
    dims = ['COMID','time'],
    coords = {'COMID':River_id,
                'time':test_span}
)

data_arrays['runoff'] = data_array




xr_dataset = xr.Dataset(data_arrays)
xr_dataset.to_zarr(store=out+"large_river.zarr", group=f'large_river', mode='w')