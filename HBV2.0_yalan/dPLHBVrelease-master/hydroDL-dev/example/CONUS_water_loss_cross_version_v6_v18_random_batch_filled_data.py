import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))

from hydroDL.model import crit, train
from hydroDL.model import rnn as rnn
from hydroDL.data import scale,PET
from hydroDL.master import loadModel
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

data_folder = "/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/gages/dataCONUS3200/"
time = pd.date_range('1980-10-01',f'1995-09-30', freq='d')
with open(data_folder+'train_data_dict.json') as f:
    train_data_dict = json.load(f)


AllTime = pd.date_range('1980-01-01', f'2020-12-31', freq='d')
index_start = AllTime.get_loc(time[0])
index_end = AllTime.get_loc(time[-1])


shapeID_str_lst= np.load("/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/shapeID_str_lst.npy")



streamflow = np.load(data_folder+"train_flow.npy")

attribute_file = '/projects/mhpi/yxs275/Data/attributes_haoyu/attributes_haoyu.csv'
attributeALL_df = pd.read_csv(attribute_file,index_col=0)
attributeALL_df = attributeALL_df.sort_values(by='id')


gage_info_file_selected_from_merit = "/projects/mhpi/data/MERIT/gage_information/formatted_gage_csvs/gages_3000_merit_info.csv"
gage_info_from_merit = pd.read_csv(gage_info_file_selected_from_merit)


gage_info_from_merit = gage_info_from_merit.sort_values(by='STAID')
gageIDs_from_merit = gage_info_from_merit['STAID'].values

attributeALL_df  = attributeALL_df[attributeALL_df['id'].isin(gageIDs_from_merit)]


attributeAllLst = attributeALL_df.columns

basin_area = np.expand_dims(attributeALL_df["area"].values,axis = 1)  ## Basin area for unit convert


lat =  attributeALL_df["lat"].values
idLst_new = attributeALL_df["id"].values
idLst_old = [int(id) for id in shapeID_str_lst]
[C, ind1, SubInd] = np.intersect1d(idLst_new, idLst_old, return_indices=True)

streamflow = streamflow[SubInd,:,:]
if(not (idLst_new==np.array(idLst_old)[SubInd]).all()):
   raise Exception("Ids of subset gage do not match with id in the attribtue file")

log_norm_cols = []
# log_norm_cols=['prcp', 'pr', 'total_precipitation', 'pre', 'potential_evaporation', 'LE',
#                    'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'ET_water', 'ET_sum', 'susm', 'smp', 'ssma', 'susma',
#                    'usgsFlow', 'streamflow', 'qobs']

attributeLst = ['ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']

attributeLst_water_loss = attributeLst.copy()
attributeLst_water_loss.append('uparea')

attributeLst_old = attributeLst.copy()


forcing_list_water_loss = ['P','Temp','PET']



key_info = [str(x) for x in idLst_new]



with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_info.json') as f:
    area_info = json.loa = '/projects/mhpi/yxs275/Data/merit_data_in_gages_3000_merit_filled_train/CONUS2800'

with open('/projects/mhpi/yxs275/Data/merit_data_in_gages_3000_merit_filled_train/' + 'merit_idx.json') as f:
    merit_idx = json.load(f)


root_zone = zarr.open_group(save_path_merit_path, mode = 'r')
Merit_all = root_zone['COMID'][:]
xTrain2 = np.full((len(Merit_all),len(time),len(forcing_list_water_loss)),np.nan)
attr2 = np.full((len(Merit_all),len(attributeLst_water_loss)),np.nan)


merit_time = pd.date_range('1980-10-01',f'2010-09-30', freq='d')
merit_start_idx = merit_time.get_loc(time[0])
merit_end_idx = merit_time.get_loc(time[-1])+1


for fid, foring_ in enumerate(forcing_list_water_loss):    
    xTrain2[:,:,fid] = root_zone[foring_][:,merit_start_idx:merit_end_idx]
# load attributes

for aid, attribute_ in enumerate(attributeLst_water_loss) :                                    
    attr2[:,aid] =  root_zone['attr'][attribute_][:] 

Ac_all = root_zone['attr']["uparea"][:] 
Ai_all = root_zone['attr']["catchsize"][:] 



stat_dict={}
for fid, forcing_item in enumerate(forcing_list_water_loss) :
    if forcing_item in log_norm_cols:
        stat_dict[forcing_item] = scale.cal_stat_gamma(xTrain2[:,:,fid])
    else:
        stat_dict[forcing_item] = scale.cal_stat(xTrain2[:,:,fid])



for aid, attribute_item in enumerate (attributeLst_water_loss):
    stat_dict[attribute_item] = scale.cal_stat(attr2[:,aid])


# stat_dict['uparea'] = scale.cal_stat(np.array(Ac_all))


streamflow_trans = scale._basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


    
forcing_norm2 = scale._trans_norm(
xTrain2, forcing_list_water_loss, stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)
xTrain2[xTrain2!=xTrain2]  = 0
forcing_norm2[forcing_norm2!=forcing_norm2]  = 0
attribute_norm2 = scale.trans_norm(attr2, list(attributeLst_water_loss), stat_dict, to_norm=True)
attribute_norm2[attribute_norm2!=attribute_norm2] = 0






Ninv1 = forcing_norm2.shape[-1]+attribute_norm2.shape[-1]
Ninv2 = len(attributeLst_water_loss)

EPOCH = 100 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 1
alpha = 0.25
HIDDENSIZE1 = 64
BUFFTIME = 365 # for each training sample, to use BUFFTIME days to warm up the states.
routing = True # Whether to use the routing module for simulated runoff
Nmul1 = 4 # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
drinv1 = 0.5
comprout = False # True is doing routing for each component
compwts = False # True is using weighted average for components; False is the simple mean
pcorr = None # or a list to give the range of precip correc

tdRep = [1, 13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
print("static parameter ", tdRep)
tdRepS = [str(ix) for ix in tdRep]
# ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
# Must set below ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
# If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
ETMod = True
Nfea1 = 3  # should be 13 when setting ETMod=True. 12 when ETMod=False

Nfea2 = 13  # should be 13 when setting ETMod=True. 12 when ETMod=False
HIDDENSIZE2 = 1024*4
Nmul2 = 4
dydrop = 0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
staind = -1  # which time step to use from the learned para time series for those static parameters
drinv2 = 0.5
model_name= 'waterlossv18'   ###Which water loss formula to use
model = rnn.MultiInv_HBVTDModel_water_loss_v6(ninv1 = Ninv1, nfea1 = Nfea1, nmul1 = Nmul1, hiddeninv1 = HIDDENSIZE1, ninv2 = Ninv2, nfea2 = Nfea2 , nmul2 = Nmul2,hiddeninv2 = HIDDENSIZE2, drinv1 = drinv1, drinv2 = drinv2,inittime=BUFFTIME,
                                routOpt=routing, comprout=comprout, compwts=compwts, staind=staind, tdlst=tdRep,
                                dydrop=dydrop, ETMod=ETMod,model_name = model_name)



# lossFun = crit.RmseLossComb(alpha=alpha)

#Transfer model saved in old code version to new version.

# rootOut_pre = "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v10_random_batch_filled_data/exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test/'

# testepoch_old = 38

# print("Load model from ", rootOut_pre)
# old_model = loadModel(rootOut_pre, epoch=testepoch_old)

# model.lstminv1 = old_model.lstminv1
# model.Ann = old_model.Ann
# train.saveModel(out, model, testepoch, modelName='model')





lossFun = crit.NSELossBatch(np.nanstd(streamflow_trans, axis=1))


rootOut = "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v18_random_batch_filled_data_dynamic_K0_correct_nfea2'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, f"exp_EPOCH{EPOCH}_BS{BATCH_SIZE}_RHO{RHO}_HS1{HIDDENSIZE1}_MUL1{Nmul1}_HS2{HIDDENSIZE2}_MUL2{Nmul2}_trainBuff{BUFFTIME}_test") # output folder to save results
if os.path.exists(out) is False:
    os.mkdir(out)

with open(out+'/dapengscaler_stat.json','w') as f: 
    json.dump(stat_dict, f)
 
# KFold for spatial test
# fold_number = 1
# # Define the number of folds (K)
# k = 6

# # Initialize the KFold cross-validator
# kf = KFold(n_splits=k, shuffle=True, random_state=randomseed)

# # Iterate through the folds
# for fold_number_idx, (train_index, test_index) in enumerate(kf.split(streamflow_trans)):
#     if fold_number_idx == fold_number:
#         xTrain_fold = xTrain[train_index,:,:]
#         forcing_LSTM_norm_fold = forcing_LSTM_norm[train_index,:,:]
#         streamflow_trans_fold = streamflow_trans[train_index,:,:]
#         attribute_norm_fold = attribute_norm[train_index,:]
# forcTuple_fold = [xTrain_fold,forcing_LSTM_norm_fold]

# trainedModel = train.trainModel(
#     model,
#     forcTuple_fold,
#     streamflow_trans_fold,
#     attribute_norm_fold,
#     lossFun,
#     nEpoch=EPOCH,
#     miniBatch=[BATCH_SIZE, RHO],
#     saveEpoch=saveEPOCH,
#     saveFolder=out,
#     bufftime=BUFFTIME)

testepoch = 0
# model_path = out
# print("Load model from ", model_path)
# model = loadModel(model_path, epoch=testepoch)

forcTuple = [xTrain2,None]




trainedModel = train.trainModel(
    model,
    forcTuple,
    streamflow_trans,
    None,
    lossFun,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=BUFFTIME,
    startepoch = testepoch+1,
    area_info = area_info,
    gage_key = key_info,
    z_waterloss = [forcing_norm2,attribute_norm2],
    merit_idx =merit_idx,
    water_loss_info = [Merit_all,Ac_all,Ai_all],
    maxmeritBatchSize = 10590)