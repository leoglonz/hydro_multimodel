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
Model_path =  "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v14_random_batch_filled_data_dynamic_K0/'
out = os.path.join(Model_path, "exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test")

#Path of data


#data_folder = '/projects/mhpi/yxs275/Data/zarr_merit_for_conus_1980-10-01-2010-09-30_unique/'
data_folder = '/projects/mhpi/yxs275/Data/zarr_merit_for_conus_1980-10-01-2010-09-30_filled_with_MSWEP/'
##epoch of model to use
testepoch = 100

## ids of GPUs to use
GPU_ids = [7]


## number of GPUs to use
num_gpus = len(GPU_ids)

## Path to save the results
results_savepath = "/projects/mhpi/yxs275/DM_output/" + '/dPL_local_daymet_new_attr_water_loss_v6v14_1980-2010_filled_with_MSWEP/'
if os.path.exists(results_savepath) is False:
    os.mkdir(results_savepath)

## Pick the regions/groups for forwarding

zonefileLst = glob.glob(data_folder+'forcing/'+"*")
zonelst =[x.split("/")[-1] for x in zonefileLst]


zonelst.sort()



time_range = pd.date_range(start=f'1980-10-01', end=f'2010-09-30', freq='D') 

variables_name = ['Qr', 'Q0', 'Q1', 'Q2','ET','SWE','Q_not_routed']

var_x_list = ['P','Temp','PET']
var_c_list = ['aridity', 'meanP', 'ETPOT_Hargr', 'NDVI', 'FW', 'meanslope', 'SoilGrids1km_sand', 'SoilGrids1km_clay',
           'SoilGrids1km_silt', 'glaciers', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt',
           'meanelevation', 'meanTa', 'permafrost', 'permeability',
           'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction','T_clay','T_gravel','T_sand','T_silt','Porosity','catchsize','uparea']



def forward_on_gpu(input):
    file_idx, gpu_id = input
    zone_number = zonelst[file_idx:file_idx+2]
    


    forcing_zone, attrs = ForcingReader1(
        root_path=data_folder,
        zone=zone_number,
        var_x_list=var_x_list,
        var_c_list=var_c_list,
        start=1980,
        end=2010
    ).read_data()



    region_number = zone_number[0] +'to'+zone_number[-1] 
    COMID = forcing_zone["COMID"].values

    with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_merit_in_gage_info.json') as f:
        area_merit_in_gage_info = json.load(f)


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
            
    uparea = np.expand_dims(attrs['uparea'].values, axis=-1)

    attributeLst2 = attributeLst.copy()
    attributeLst2.append('uparea')
    attribute2 = np.concatenate((attribute,uparea), axis=-1)

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
    

    attribute_norm2 = scale._trans_norm(attribute2, list(attributeLst2), stat_dict, log_norm_cols=log_norm_cols, to_norm=True)
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
    testbatch =600 #len(indexes)

    device = torch.device("cuda:" + str(gpu_id))

    modelFile = os.path.join(model_path, 'model_Ep' + str(testepoch) + '.pt')
    testmodel = torch.load(modelFile, map_location=device)

    testmodel.inittime = 0


    train.testModel_multiGPU(
            testmodel, testTuple,attribute_norm2,area_merit_in_gage_info,COMID, region_number,time_range,variables_name,results_savepath, c=None, device =  device, batchSize=testbatch)
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
        self.time_range = pd.date_range(start=f'{self.start}-10-01', end=f'{self.end}-09-30',freq = 'D')


    def read_data(self) -> xr.Dataset:

        all_forcing_ds = []
        all_attrs_ds = []
        all_COMID = []

        for zone in self.zone:  # Assuming self.zones is a list of all zones
            print(f"Processing zone: {zone}")
            forcingroot = zarr.open_group((self.root_path + 'forcing/' + zone), mode='r')
            COMID = forcingroot['COMID'][:]
            all_COMID.append(COMID)  # Collecting COMIDs from all zones

            data_arrays = {}
            start_time = time.time()
            for var_x in self.forcing:
                data = forcingroot[var_x][:]
                data_array = xr.DataArray(
                    data,
                    dims=['COMID', 'time'],
                    coords={'COMID': COMID, 'time': self.time_range}
                )
                data_arrays[var_x] = data_array

            forcing_ds = xr.Dataset(data_arrays)
            all_forcing_ds.append(forcing_ds)

            attrroot = zarr.open_group((self.root_path + 'attr/' + zone), mode='r')
            attr_arrays = {}
            for var_c in self.attr:
                attr = attrroot[var_c][:]
                c_array = xr.DataArray(
                    attr,
                    dims=['rivid'],
                    coords={'rivid': COMID}
                )
                attr_arrays[var_c] = c_array

            attrs_ds = xr.Dataset(attr_arrays)
            all_attrs_ds.append(attrs_ds)

            end_time = time.time()
            print(f'Reading zone {zone} takes {(end_time - start_time):.2f}s')

        # Concatenate all COMID arrays
        combined_COMID = np.concatenate(all_COMID)
        
        # Use xr.concat to concatenate datasets along a new dimension, aligning by 'COMID' using combined_COMID
        combined_forcing_ds = xr.concat(all_forcing_ds, dim=pd.Index(combined_COMID, name='COMID'))
        combined_attrs_ds = xr.concat(all_attrs_ds, dim=pd.Index(combined_COMID, name='rivid'))

        return combined_forcing_ds, combined_attrs_ds



#main parallel code

startTime = time.time()

for x in range(0,len(zonelst),2):

    forward_on_gpu((x,GPU_ids[0]))


# items = [x for x in range(len(subzone_lst))]
# GPU_ids_list  = [GPU_ids[x % len(GPU_ids)] for x in items]

# processeornumber = num_gpus*2
# iS = np.arange(0, len(items), processeornumber)
# iE = np.append(iS[1:], len(items))

# for i in range(len(iS)):
#     subGPU_ids_list = GPU_ids_list[iS[i]:iE[i]]
#     subitem = items[iS[i]:iE[i]]

#     pool = multiprocessing.Pool(processes=num_gpus*2)

#     print("Will working zone ", subzone_lst[iS[i]:iE[i]])
#     print("zone idx ", subitem)
#     print("GPUs ", subGPU_ids_list)
#     results = pool.imap(forward_on_gpu, ((subitem[idx], gpuid,) for idx, gpuid in enumerate(subGPU_ids_list)))

#     pool.close()
#     pool.join()

print("Cost time: ", time.time() - startTime  )