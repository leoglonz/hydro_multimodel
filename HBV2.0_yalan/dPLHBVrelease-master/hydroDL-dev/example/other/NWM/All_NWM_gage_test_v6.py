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
rootOut = "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v18_random_batch_filled_data_dynamic_K0_correct_nfea2/'
out = os.path.join(rootOut, "exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test/")  # output folder to save results



all_time_range = pd.date_range(start=f'{1980}-01-01', end=f'{2020}-12-31', freq='D')

start_year = 1980
end_year = 2000
# start_year = 1997
# end_year = 2020

warmup_span = pd.date_range(f'{start_year}-01-01',f'{start_year+1}-12-31', freq='d')

test_span = pd.date_range(f'{start_year}-01-01',f'{end_year}-12-31', freq='d')

with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_info_all_gages.json') as f:
    area_info_all_gages = json.load(f)


NWM_gage_folder  = glob.glob('/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/daily_simulation_updated'+"/*")
NWM_gage_all =[x.split("/")[-1] for x in NWM_gage_folder]

NWM_gage_all = [str(int(x)) for x in NWM_gage_all]

NWM_gage = [x for x in NWM_gage_all if x in area_info_all_gages.keys()]



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
       'snowfall_fraction','uparea']

forcing_list_water_loss = ['P','Temp','PET']

save_path_merit_path = '/projects/mhpi/yxs275/Data/merit_data_in_NWM_gage_test/NWM_gage'

with open('/projects/mhpi/yxs275/Data/merit_data_in_NWM_gage_test/' + 'merit_idx.json') as f:
    merit_idx = json.load(f)


root_zone = zarr.open_group(save_path_merit_path, mode = 'r')
Merit_all = root_zone['COMID'][:]
xTrain2 = np.full((len(Merit_all),len(test_span),len(forcing_list_water_loss)),np.nan)
attr2 = np.full((len(Merit_all),len(attributeLst_water_loss)),np.nan)



merit_start_idx = all_time_range.get_loc(test_span[0])
merit_end_idx = all_time_range.get_loc(test_span[-1])+1

for fid, foring_ in enumerate(forcing_list_water_loss):    
    xTrain2[:,:,fid] = root_zone[foring_][:,merit_start_idx:merit_end_idx]
# load attributes

for aid, attribute_ in enumerate(attributeLst_water_loss) :                                    
    attr2[:,aid] =  root_zone['attr'][attribute_][:] 

Ac_all = root_zone['attr']["uparea"][:] 
Ai_all = root_zone['attr']["catchsize"][:] 






with open(out + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)

forcing_norm2 = scale._trans_norm(
xTrain2, forcing_list_water_loss, stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)
xTrain2[xTrain2!=xTrain2]  = 0
forcing_norm2[forcing_norm2!=forcing_norm2]  = 0

attribute_norm2 = scale.trans_norm(attr2, list(attributeLst_water_loss), stat_dict, to_norm=True)
attribute_norm2[attribute_norm2!=attribute_norm2] = 0


xTrain2 = np.concatenate((xTrain2[:,:len(warmup_span),:],xTrain2),axis = 1)

forcing_norm2 = np.concatenate((forcing_norm2[:,:len(warmup_span),:],forcing_norm2),axis = 1)

testTuple = [xTrain2,None]





testepoch = 100
model_path = out
print("Load model from ", model_path)
testmodel = loadModel(model_path, epoch=testepoch)

testbatch =1500 #len(indexes)


filePathLst = [out+f"/NWM_gage_Qs_{start_year}_{end_year}_warmuped.csv",out+f"/NWM_gage_Q0_{start_year}_{end_year}_warmuped.csv",out+f"/NWM_gage_Q1_{start_year}_{end_year}_warmuped.csv",out+f"/NWM_gage_Q2_{start_year}_{end_year}_warmuped.csv",out+f"/NWM_gage_ET_{start_year}_{end_year}_warmuped.csv",out+f"/NWM_gage_SWE_{start_year}_{end_year}_warmuped.csv",out+f"/NWM_gage_Qpp_{start_year}_{end_year}_warmuped.csv"]


testmodel.inittime = 0

train.testModel_merit(
    testmodel, testTuple, [forcing_norm2,attribute_norm2], c=None,water_loss_info=[Merit_all,Ac_all,Ai_all] ,  area_info = area_info_all_gages,gage_key = NWM_gage,  batchSize=testbatch, filePathLst=filePathLst)




variables_name = ['Qs', 'Q0', 'Q1', 'Q2','ET','SWE','Qpp']

data_arrays = {}
for idx, var_x in enumerate(variables_name):
    dataPred = pd.read_csv(  out+f"/NWM_gage_{var_x}_{start_year}_{end_year}_warmuped.csv", dtype=np.float32, header=None).values
    dataPred = np.expand_dims(dataPred, axis=-1)
    data_array = xr.DataArray(
        dataPred[:,len(warmup_span):,0],
       # dataPred[:,:,0],
        dims = ['COMID','time'],
        coords = {'COMID':NWM_gage,
                    'time':test_span}
    )

    data_arrays[var_x] = data_array


xr_dataset = xr.Dataset(data_arrays)
xr_dataset.to_zarr(store=out, group=f'NWM_gage_simulation_{start_year}_{end_year}_warmuped', mode='w')
