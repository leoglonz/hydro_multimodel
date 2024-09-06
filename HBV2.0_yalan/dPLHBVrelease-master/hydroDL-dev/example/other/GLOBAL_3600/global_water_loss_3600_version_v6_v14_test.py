import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))

from hydroDL.model import crit, train
from hydroDL.model import rnn as rnn
from hydroDL.data import scale,PET
from hydroDL.master import loadModel
from hydroDL.post import plot, stat
import torch.nn.functional as F
import os
import numpy as np
import torch
import random
import json
import pandas as pd
import json
import zarr
from sklearn.model_selection import KFold
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
traingpuid = 7
torch.cuda.set_device(traingpuid)

data_folder = "/projects/mhpi/data/merit_data_in_global_gages/organized_data/"
totaltime = pd.date_range('1979-01-01', '2019-12-31', freq='d')

testingtime = pd.date_range('1980-01-01', '1997-12-31', freq='d') 
test_start_idx = totaltime.get_loc(testingtime[0])
test_end_idx = totaltime.get_loc(testingtime[-1])+1


streamflow_test = np.load(data_folder+"water_loss_data/runoff_global_3600.npy")[:,test_start_idx:test_end_idx]


trainingtime = pd.date_range('1999-01-01', '2016-12-31', freq='d') 
train_start_idx = totaltime.get_loc(trainingtime[0])
train_end_idx = totaltime.get_loc(trainingtime[-1])+1

streamflow_train = np.load(data_folder+"water_loss_data/runoff_global_3600.npy")[:,train_start_idx:train_end_idx]

streamflow = np.concatenate((streamflow_train,streamflow_test), axis = 1)

with open(data_folder +'water_loss_data/area_info.json') as f:
    key_info = json.load(f)


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


with open(data_folder + 'merit_idx.json') as f:
    merit_idx = json.load(f)


root_zone = zarr.open_group(data_folder + 'GLOBAL3600', mode = 'r')
Merit_all = root_zone['COMID'][:]

xTrain2_test = np.full((len(Merit_all),len(trainingtime),len(forcing_list_water_loss)),np.nan)
xTrain2_train = np.full((len(Merit_all),len(trainingtime),len(forcing_list_water_loss)),np.nan)
attr2 = np.full((len(Merit_all),len(attributeLst_water_loss)),np.nan)


merit_time = pd.date_range('1980-01-01',f'2020-12-31', freq='d')
merit_start_idx_train = merit_time.get_loc(trainingtime[0])
merit_end_idx_train = merit_time.get_loc(trainingtime[-1])+1
merit_start_idx_test = merit_time.get_loc(testingtime[0])
merit_end_idx_test = merit_time.get_loc(testingtime[-1])+1

for fid, foring_ in enumerate(forcing_list_water_loss):    
    xTrain2_train[:,:,fid] = root_zone[foring_][:,merit_start_idx_train:merit_end_idx_train]
    xTrain2_test[:,:,fid] = root_zone[foring_][:,merit_start_idx_test:merit_end_idx_test]
# load attributes

for aid, attribute_ in enumerate(attributeLst_water_loss) :                                    
    attr2[:,aid] =  root_zone['attr'][attribute_][:] 

Ac_all = root_zone['attr']["uparea"][:] 
Ai_all = root_zone['attr']["catchsize"][:] 


xTrain2 =  np.concatenate((xTrain2_train,xTrain2_test), axis = 1)

rootOut = "/projects/mhpi/yxs275/model/globalModel/"+'/dPL_local_daymet_new_attr_water_loss_v6v14_random_batch_filled_data_dynamic_K0'

out = os.path.join(rootOut, f"exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test") # output folder to save results


with open(out + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)

with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_info_GLOBAL4000.json') as f:
    area_info = json.load(f)




forcing_norm2 = scale._trans_norm(
xTrain2, forcing_list_water_loss, stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)
xTrain2[xTrain2!=xTrain2]  = 0
forcing_norm2[forcing_norm2!=forcing_norm2]  = 0
attribute_norm2 = scale.trans_norm(attr2, list(attributeLst_water_loss), stat_dict, to_norm=True)
attribute_norm2[attribute_norm2!=attribute_norm2] = 0




testTuple = [xTrain2,None]





testepoch = 100
model_path = out
print("Load model from ", model_path)
testmodel = loadModel(model_path, epoch=testepoch)

testbatch =1000 #len(indexes)

filePathLst = [out+"/Qs.csv",out+"/Q0.csv",out+"/Q1.csv",out+"/Q2.csv",out+"/ET.csv",out+"/SWE.csv",out+"/Qpp.csv"]



testmodel.inittime = 0

# train.testModel_merit(
#     testmodel, testTuple, [forcing_norm2,attribute_norm2], c=None,water_loss_info=[Merit_all,Ac_all,Ai_all] ,  area_info = area_info,gage_key = key_info,  batchSize=testbatch, filePathLst=filePathLst)


dataPred = pd.read_csv(  out+"/Qs.csv", dtype=np.float32, header=None).values
dataPred = np.expand_dims(dataPred, axis=-1)

warmup_for_testing = len(trainingtime)
evaDict = [stat.statError(dataPred[:,warmup_for_testing:,0], streamflow[:,warmup_for_testing:,0])]
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