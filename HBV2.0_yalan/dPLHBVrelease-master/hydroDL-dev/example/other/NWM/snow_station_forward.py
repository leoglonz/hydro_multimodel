import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))

from hydroDL.model import train
from hydroDL.data import scale,PET
import os
import numpy as np
import torch
import random
import pandas as pd
import json
import multiprocessing
import glob
import time
import xarray as xr
import zarr
#from tqdm import tqdm
## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Path of model
Model_path =  "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v14_random_batch_filled_data_dynamic_K0/'
out = os.path.join(Model_path, "exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test")

#Path of data
SWE_crd = pd.read_csv("/projects/mhpi/data/NWM/snow/SNOTEL_filter_data_1988/crd/crd.csv")
lat_snotel = SWE_crd['lat'].values
lon_snotel = SWE_crd['lon'].values

snotel_time_range = pd.date_range(start=f'{1987}-10-01', end=f'{2020}-12-31', freq='D')


snotel_SWE_path = '/projects/mhpi/data/NWM/snow/SNOTEL_filter_data_1988/'
with open(snotel_SWE_path + '/SWE_Station.json') as f:
    MERITinfo_dict = json.load(f)

MERIT_lat = MERITinfo_dict['lat']
MERIT_lon = MERITinfo_dict['lon']
MERIT_COMID = MERITinfo_dict['COMID']
MERIT_COMID = [str(int(x)) for x in MERIT_COMID]

new_MERIT_COMID = []
for idx in range(len(lat_snotel)):
    id = np.where((np.array(MERIT_lat) == lat_snotel[idx]) & (np.array(MERIT_lon) == lon_snotel[idx]))[0][0]
    new_MERIT_COMID.append(MERIT_COMID[id])


var_x_list = ['P','Temp','PET']
var_c_list = ['ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
                    'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
                    'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
                    'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
                    'meanelevation', 'meanslope', 'permafrost', 'permeability',
                    'seasonality_P', 'seasonality_PET', 'snow_fraction',
                    'snowfall_fraction']

dHBV_time_range = pd.date_range(start=f'{1980}-01-01', end=f'{2020}-12-31', freq='D')
dHBV_start_id = dHBV_time_range.get_loc(snotel_time_range[0])
dHBV_end_id = dHBV_time_range.get_loc(snotel_time_range[-1])+1

forcing = np.full((len(lat_snotel),len(snotel_time_range),len(var_x_list)),np.nan)
attr = np.full((len(lat_snotel),len(var_c_list)),np.nan)
#data_folder = '/projects/mhpi/yxs275/Data/zarr_merit_for_conus_1980-10-01-2010-09-30_unique/'
forcing_data_folder = '/projects/mhpi/data/conus/zarr/'
##epoch of model to use
testepoch = 100

## ids of GPUs to use
gpu_id =3






## Path to save the results
results_savepath = '/projects/mhpi/data/NWM/snow/'
if os.path.exists(results_savepath) is False:
    os.mkdir(results_savepath)

## Pick the regions/groups for forwarding

subzonefile_lst = []
zonefileLst = glob.glob(forcing_data_folder+"*")
zonelst =[int(x.split("/")[-1]) for x in zonefileLst]

zonelst.sort()
for largezonefile in zonefileLst:
    subzonefile_lst.extend(glob.glob(largezonefile+"/*"))
subzonefile_lst.sort()




attributes_points = pd.read_csv("/projects/mhpi/data/NWM/snow/attribute_point.csv")
Snotel_attr = pd.read_csv("/projects/mhpi/data/NWM/snow/SNOTEL_filter_data_1988/attributes_all.csv")

int_new_MERIT_COMID = [int(x) for x in new_MERIT_COMID]


for idx in range(len(subzonefile_lst)):

    print("Working on zone ", subzonefile_lst[idx])

    forcing_root_zone = zarr.open_group(subzonefile_lst[idx], mode = 'r')
    
    AORC_forcing_root_zone = zarr.open_group(subzonefile_lst[idx], mode = 'r')
    gage_COMIDs = forcing_root_zone['COMID'][:]

    [C, ind1, SubInd] = np.intersect1d(int_new_MERIT_COMID, gage_COMIDs, return_indices=True)
    
    if SubInd.any():

        for variablei, variable in enumerate(var_x_list):
            forcing[ind1,:,variablei] = forcing_root_zone[variable][SubInd,dHBV_start_id:dHBV_end_id]

        for variablei, variable in enumerate(var_c_list):
            attr[ind1,variablei]= forcing_root_zone['attrs'][variable][SubInd]

for variablei, variable in enumerate(var_c_list):
    if variable =='meanelevation':
        attr[:,variablei] = Snotel_attr['mean_elev'].values
    else:
        attr[:,variablei] = attributes_points[variable].values

site_daymet_forcing = pd.read_csv('/projects/mhpi/data/NWM/snow/prcp.csv').values[:,2:][:,dHBV_start_id:dHBV_end_id]

site_daymet_tmin = pd.read_csv('/projects/mhpi/data/NWM/snow/tmin.csv').values[:,2:][:,dHBV_start_id:dHBV_end_id]
site_daymet_tmax = pd.read_csv('/projects/mhpi/data/NWM/snow/tmax.csv').values[:,2:][:,dHBV_start_id:dHBV_end_id]


snotel_forcing_path = '/projects/mhpi/data/NWM/snow/'

for year in range(2001, 2020):
    SWE_year = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    start_id = snotel_time_range.get_loc(SWE_year[0])
    end_id = snotel_time_range.get_loc(SWE_year[-1])+1
    precip_data = pd.read_csv(snotel_forcing_path+'pr_insitu/'+f'{year}/'+'pr_insitu.csv',skiprows = 0,header=None).values*1000
    tmin_data = pd.read_csv(snotel_forcing_path+'tmmn_insitu/'+f'{year}/'+'tmmn_insitu.csv',skiprows = 0,header=None).values-273.15
    tmax_data = pd.read_csv(snotel_forcing_path+'tmmx_insitu/'+f'{year}/'+'tmmx_insitu.csv',skiprows = 0,header=None).values-273.15
    site_daymet_forcing[:,start_id:end_id] = precip_data
    site_daymet_tmin[:,start_id:end_id][np.where(tmin_data==tmin_data)] = tmin_data[np.where(tmin_data==tmin_data)]
    site_daymet_tmax[:,start_id:end_id][np.where(tmax_data==tmax_data)] = tmax_data[np.where(tmax_data==tmax_data)]


time_range = snotel_time_range

variables_name = ['Qr', 'Q0', 'Q1', 'Q2','ET','SWE','Q_not_routed']

Tex = [19871001, 20201231]

tmin = np.swapaxes(site_daymet_tmin, 0,1)
tmax = np.swapaxes(site_daymet_tmax, 0,1)

tmean = (tmin+tmax)/2

latarray = np.tile(lat_snotel, [tmin.shape[0], 1])
pet = PET.hargreaves(tmin, tmax, tmean, lat=latarray, trange=Tex)

forcing[:,:,0] = site_daymet_forcing
forcing[:,:,1] = np.swapaxes(tmean, 0,1)
forcing[:,:,2] = np.swapaxes(pet, 0,1)



region_number = 'snotel_point'
COMID = new_MERIT_COMID

with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_all_merit_info.json') as f:
    area_all_merit_info = json.load(f)

attributeLst = ['ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
                'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
                'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
                'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
                'meanelevation', 'meanslope', 'permafrost', 'permeability',
                'seasonality_P', 'seasonality_PET', 'snow_fraction',
                'snowfall_fraction','uparea']


uparea_zone = [area_all_merit_info[x]['uparea'][0] for x in COMID]

# load forcing
xTrain = forcing
# load attributes
uparea_zone = np.expand_dims(np.array(uparea_zone), axis=-1)
attribute = np.concatenate((attr,uparea_zone), axis=-1)


xTrain = scale.fill_Nan(xTrain)
log_norm_cols = []
# log_norm_cols=['P','prcp', 'pr', 'total_precipitation', 'pre',  'LE',
#                    'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'susm', 'smp', 'ssma', 'susma',
#                    'usgsFlow', 'streamflow', 'qobs']

## Input normalization
with open(out + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)
#stat_dict["catchsize"] = stat_dict["area"]
forcing_LSTM_norm = scale._trans_norm(
    xTrain.copy(), var_x_list, stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)
forcing_LSTM_norm [forcing_LSTM_norm != forcing_LSTM_norm] = 0
xTrain[xTrain != xTrain] = 0


attribute_norm2 = scale._trans_norm(attribute, list(attributeLst), stat_dict, log_norm_cols=log_norm_cols, to_norm=True)
attribute_norm2[attribute_norm2 != attribute_norm2] = 0


attribute_norm2_expand = np.expand_dims(attribute_norm2, axis=1)
attribute_norm2_expand = np.repeat(attribute_norm2_expand, forcing_LSTM_norm.shape[1], axis=1)




# load the model

model_path = out
print("Load model from ", model_path)


#zTest = forcing_LSTM_norm  # Add attributes to historical forcings as the inversion part
zTest = np.concatenate([forcing_LSTM_norm, attribute_norm2_expand], 2)
xTest = xTrain
testTuple = (xTest[:,:,:], zTest[:,:,:])
testbatch =300 #len(indexes)

device = torch.device("cuda:" + str(gpu_id))

modelFile = os.path.join(model_path, 'model_Ep' + str(testepoch) + '.pt')
testmodel = torch.load(modelFile, map_location=device)

testmodel.inittime = 0


train.testModel_multiGPU(
        testmodel, testTuple,attribute_norm2,area_all_merit_info,COMID, region_number,time_range,variables_name,results_savepath, c=None, device =  device, batchSize=testbatch)
print(f"Group/region {region_number} is done on GPU {gpu_id}")



