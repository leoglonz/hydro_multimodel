import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))

from hydroDL.model import train
from hydroDL.data import scale
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
Model_path =  "/projects/mhpi/yxs275/model/water_loss_model/" + '/dPL_local_daymet_new_attr_water_loss_v3_correct_Ai/'
out = os.path.join(Model_path, "exp_EPOCH50_BS100_RHO365_HS512_trainBuff365")

#Path of data

data_folder = '/projects/mhpi/yxs275/Data/zarr_zone/'
##epoch of model to use
testepoch = 50

## ids of GPUs to use
GPU_ids = [4]


## number of GPUs to use
num_gpus = len(GPU_ids)

## Path to save the results
results_savepath = "/projects/mhpi/yxs275/DM_output/water_loss_model/" 
if os.path.exists(results_savepath) is False:
    os.mkdir(results_savepath)
results_savepath = results_savepath+ '/dPL_local_daymet_new_attr_water_loss_v3_correct_Ai_all_merit/'
if os.path.exists(results_savepath) is False:
    os.mkdir(results_savepath)

## Pick the regions/groups for forwarding
subzonefile_lst = []
zonefileLst = glob.glob(data_folder+"*")
zonelst =[int(x.split("/")[-1]) for x in zonefileLst]

zonelst.sort()
for largezonefile in zonefileLst:
    subzonefile_lst.extend(glob.glob(largezonefile+"/*"))
subzonefile_lst.sort()

#subzone_lst =[x.split("/")[-1] for x in subzonefile_lst]

existing_files  = glob.glob(results_savepath+"/*")
existing_subzone =[x.split("/")[-1] for x in existing_files]

subzonefile_lst_new = []
for file in subzonefile_lst:
    zid =  file.split("/")[-1]
    if zid not in existing_subzone:
        subzonefile_lst_new.append(file)

subzone_lst_new  =[x.split("/")[-1] for x in subzonefile_lst_new]


startyear = 1980
endyear = 2020
time_range = pd.date_range(start=f'{startyear}-01-01', end=f'{endyear}-12-31', freq='D')

variables_name = ['Qr', 'Q0', 'Q1', 'Q2','ET']

var_x_list = ['P','Temp','PET']
var_c_list = ['aridity', 'meanP', 'ETPOT_Hargr', 'NDVI', 'FW', 'meanslope', 'SoilGrids1km_sand', 'SoilGrids1km_clay',
           'SoilGrids1km_silt', 'glaciers', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt',
           'meanelevation', 'meanTa', 'permafrost', 'permeability',
           'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction','T_clay','T_gravel','T_sand','T_silt','Porosity','catchsize']



def forward_on_gpu(input):
    file_idx, gpu_id = input
    zone_number = subzone_lst_new[file_idx]
    data_folder = subzonefile_lst_new[file_idx].split(f"{zone_number}")[0]


    forcing_zone, attrs = ForcingReader1(
        root_path=data_folder,
        zone=zone_number,
        var_x_list=var_x_list,
        var_c_list=var_c_list,
        start=1980,
        end=2020
    ).read_data()



    region_number = zone_number
    COMID = forcing_zone["COMID"].values
    COMID = [str(int(x)) for x in COMID]
    with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_all_merit_info.json') as f:
        area_all_merit_info = json.load(f)



    attributeLst = ['ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
                    'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
                    'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
                    'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
                    'meanelevation', 'meanslope', 'permafrost', 'permeability',
                    'seasonality_P', 'seasonality_PET', 'snow_fraction',
                    'snowfall_fraction']


    # load forcing
    for fid, foring_ in enumerate(var_x_list):
        foring_data = np.expand_dims(forcing_zone[foring_].values, axis=-1)
        if fid==0:
            xTrain = foring_data
        else:
            xTrain = np.concatenate((xTrain,foring_data), axis=-1)
    # load attributes
    for aid, attribute_ in enumerate(attributeLst) :
        attribute_data = np.expand_dims(attrs[attribute_].values, axis=-1)
        if aid==0:
            attribute = attribute_data
        else:
            attribute = np.concatenate((attribute,attribute_data), axis=-1)



    xTrain = scale.fill_Nan(xTrain)

    log_norm_cols=['prcp', 'pr', 'total_precipitation', 'pre',  'LE',
                       'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'susm', 'smp', 'ssma', 'susma',
                       'usgsFlow', 'streamflow', 'qobs']

    ## Input normalization
    with open(out + '/dapengscaler_stat.json') as f:
        stat_dict = json.load(f)
    #stat_dict["catchsize"] = stat_dict["area"]
    forcing_LSTM_norm = scale._trans_norm(
        xTrain.copy(), ['prcp','tmean','pet'], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
    )
    forcing_LSTM_norm [forcing_LSTM_norm != forcing_LSTM_norm] = 0
    xTrain[xTrain != xTrain] = 0
    attribute_norm = scale._trans_norm(attribute, list(attributeLst), stat_dict, log_norm_cols=log_norm_cols, to_norm=True)
    attribute_norm[attribute_norm != attribute_norm] = 0
    attribute_norm = np.expand_dims(attribute_norm, axis=1)
    attribute_norm = np.repeat(attribute_norm, forcing_LSTM_norm.shape[1], axis=1)

    # load the model

    model_path = out
    print("Load model from ", model_path)


    zTest = np.concatenate([forcing_LSTM_norm, attribute_norm], 2)  # Add attributes to historical forcings as the inversion part
    xTest = xTrain
    testTuple = (xTest[:,:,:], zTest[:,:,:])
    testbatch =200 #len(indexes)

    device = torch.device("cuda:" + str(gpu_id))

    modelFile = os.path.join(model_path, 'model_Ep' + str(testepoch) + '.pt')
    testmodel = torch.load(modelFile, map_location=device)

    testmodel.inittime = 0


    train.testModel_multiGPU(
            testmodel, testTuple,area_all_merit_info,COMID, region_number,time_range,variables_name,results_savepath, c=None, device =  device, batchSize=testbatch)
    print(f"Group/region {zone_number} is done on GPU {gpu_id}")





class ForcingReader1():
    def __init__(self,root_path,zone,var_x_list,var_c_list,start,end):
        self.root_path = root_path
        self.zone = zone
        self.forcing = var_x_list
        self.attr = var_c_list
        self.continent = str(zone)[0]
        self.start = start
        self.end = end
        self.time_range = pd.date_range(start=f'{self.start}-01-01', end=f'{self.end}-12-31',freq = 'D')


    def read_data(self) -> xr.Dataset:

        root = zarr.open_group(self.root_path, mode = 'r')
        zone_group = f'{self.zone}'
        COMID = root[zone_group]['COMID'][:]

        data_arrays = {}
        #data_x = np.full((COMID.shape[0],self.time_range.shape[0],len(self.forcing)),np.nan)
        start_time = time.time()
        for var_x in self.forcing:
            data = root[zone_group][var_x][:]
            data_array = xr.DataArray(
                data,
                dims = ['COMID','time'],
                coords = {'COMID':COMID,
                          'time':self.time_range}
            )

            data_arrays[var_x] = data_array

        forcing_ds = xr.Dataset(data_arrays)


        attr_arrays = {}
        for var_c in self.attr:
            attr = root[zone_group]['attrs'][var_c][:]
            c_array = xr.DataArray(
                attr,
                dims = ['rivid'],
                coords = {'rivid':COMID}
            )

            attr_arrays[var_c] = c_array

        attrs_ds = xr.Dataset(attr_arrays)
        end_time = time.time()
        print(f'reading zone {self.zone} takes {(end_time - start_time):.2f}s')

        return forcing_ds,attrs_ds



#main parallel code

startTime = time.time()


# forward_on_gpu((6,1))


items = [x for x in range(len(subzone_lst_new))]
GPU_ids_list  = [GPU_ids[x % len(GPU_ids)] for x in items]

processeornumber = num_gpus*2
iS = np.arange(0, len(items), processeornumber)
iE = np.append(iS[1:], len(items))

for i in range(len(iS)):
    subGPU_ids_list = GPU_ids_list[iS[i]:iE[i]]
    subitem = items[iS[i]:iE[i]]

    pool = multiprocessing.Pool(processes=num_gpus*2)

    print("Will working zone ", subzone_lst_new[iS[i]:iE[i]])
    print("zone idx ", subitem)
    print("GPUs ", subGPU_ids_list)
    results = pool.imap(forward_on_gpu, ((subitem[idx], gpuid,) for idx, gpuid in enumerate(subGPU_ids_list)))

    pool.close()
    pool.join()

print("Cost time: ", time.time() - startTime  )
