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
traingpuid = 2
torch.cuda.set_device(traingpuid)

NWM_cal_basin_path = "/projects/mhpi/hjj5218/data/NWM/zarr/"


NWM_cal_basin_files  = glob.glob(NWM_cal_basin_path+"/*")
NWM_cal_basin_gage =[x.split("/")[-1] for x in NWM_cal_basin_files]
NWM_cal_basin_gage.sort()


zarr_save_path = '/projects/mhpi/yxs275/Data/GAGES_2/zarr/'


all_time_range = pd.date_range(start=f'{1980}-01-01', end=f'{2019}-12-31', freq='D')




with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_info_all_gages.json') as f:
    area_info_all_gages = json.load(f)


training_timespan = pd.date_range('1980-10-01',f'2000-09-30', freq='d')

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


forcing_start_id = all_time_range.get_loc(training_timespan[0])
forcing_end_id = all_time_range.get_loc(training_timespan[-1])+1

trained_foring_selected = np.full((len(NWM_cal_basin_gage),len(training_timespan),len(var_x_list)),np.nan)

trained_attr_selected = np.full((len(NWM_cal_basin_gage),len(attributeLst)),np.nan)
for gageidx, gage in enumerate(NWM_cal_basin_gage):
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
forcing_train = trained_foring_selected[:,:,:-1]


streamflow_trans = scale._basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


attr_train = trained_attr_selected

log_norm_cols=[]



stat_dict={}
for fid, forcing_item in enumerate(var_x_list[:-1]) :
    if forcing_item in log_norm_cols:
        stat_dict[forcing_item] = scale.cal_stat_gamma(forcing_train[:,:,fid])
    else:
        stat_dict[forcing_item] = scale.cal_stat(forcing_train[:,:,fid])



for aid, attribute_item in enumerate (attributeLst):
    stat_dict[attribute_item] = scale.cal_stat(attr_train[:,aid])



forcing_LSTM_norm = scale._trans_norm(
    forcing_train.copy(), var_x_list[:-1], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)

forcing_train[forcing_train!=forcing_train]  = 0

forcing_LSTM_norm[forcing_LSTM_norm!=forcing_LSTM_norm]  = 0

attribute_norm = scale.trans_norm(attr_train, list(attributeLst), stat_dict, to_norm=True)
attribute_norm[attribute_norm!=attribute_norm] = 0



Ninv = forcing_LSTM_norm.shape[-1]+attribute_norm.shape[-1]
EPOCH = 100 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 5
alpha = 0.25
HIDDENSIZE = 256
BUFFTIME = 365 # for each training sample, to use BUFFTIME days to warm up the states.
routing = True # Whether to use the routing module for simulated runoff
Nmul = 16 # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
comprout = False # True is doing routing for each component
compwts = False # True is using weighted average for components; False is the simple mean
pcorr = None # or a list to give the range of precip correc

tdRep = [1,3, 13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
tdRepS = [str(ix) for ix in tdRep]
# ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
# Must set below ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
# If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
ETMod = True
Nfea = 14  # should be 13 when setting ETMod=True. 12 when ETMod=False
dydrop = 0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
staind = -1  # which time step to use from the learned para time series for those static parameters

model = rnn.MultiInv_HBVTDModel(ninv=Ninv, nfea=Nfea, nmul=Nmul, hiddeninv=HIDDENSIZE, inittime=BUFFTIME,
                                routOpt=routing, comprout=comprout, compwts=compwts, staind=staind, tdlst=tdRep,
                                dydrop=dydrop, ETMod=ETMod)


lossFun = crit.NSELossBatch(np.nanstd(streamflow_trans, axis=1))

rootOut = "/projects/mhpi/yxs275/model/NWM"+'/dHBV_NWM_calibration_basins_NSE_wo_log_with_cr_dy_K0_spatial_daymet/'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, f"exp_EPOCH{EPOCH}_BS{BATCH_SIZE}_RHO{RHO}_HS{HIDDENSIZE}_trainBuff{BUFFTIME}") # output folder to save results
if os.path.exists(out) is False:
    os.mkdir(out)

with open(out+'/dapengscaler_stat.json','w') as f:
    json.dump(stat_dict, f)


testepoch = 20

model_path = out
print("Load model from ", model_path)
model = loadModel(model_path, epoch=testepoch)


forcTuple = [forcing_train,forcing_LSTM_norm]
trainedModel = train.trainModel(
    model,
    forcTuple,
    streamflow_trans,
    attribute_norm,
    lossFun,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=BUFFTIME,
    startepoch = testepoch+1,)