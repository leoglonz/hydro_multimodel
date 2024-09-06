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
rootOut = "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v14_random_batch_filled_data_dynamic_K0/'
out = os.path.join(rootOut, "exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test/")  # output folder to save results
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
       'snowfall_fraction','uparea']

forcing_list_water_loss = ['P','Temp','PET']


with open('/projects/mhpi/hjj5218/data/Main_River/CONUS_v3/'+'area_info_main_river.json') as f:
    area_info = json.load(f)

bad_rivers = ["08092600","13171620","13290200", "09520700"]
for item in bad_rivers:
    area_info.pop(item)

attribut_river_file = "/projects/mhpi/hjj5218/data/Main_River/CONUS_v2/main_river_41.csv"
attribut_river = pd.read_csv(attribut_river_file)
area_river = attribut_river['area'].values

obs_ft = np.load('/projects/mhpi/hjj5218/data/Main_River/CONUS_v2/obs_streamflow_1980_2020.npy')/0.0283168 

obs_ft = np.swapaxes(obs_ft,1,0)
obs_mm_day = scale._basin_norm(
                        np.expand_dims(obs_ft,axis = -1 ) ,  np.expand_dims(area_river,axis = -1), to_norm=True
                    )  ## from ft^3/s to mm/day




zarr_save_path = '/projects/mhpi/hjj5218/data/Main_River/CONUS_v2/distributed.zarr'
Largeriverfiles  = glob.glob(zarr_save_path+"/*")
Largerive_gage =[x.split("/")[-1] for x in Largeriverfiles]
Largerive_gage.sort()

dHBV_time_range_all = pd.date_range(start=f'{1980}-01-01', end=f'{2020}-12-31', freq='D')

warmup_span = pd.date_range('1982-10-01',f'1985-09-30', freq='d')
test_span = pd.date_range('1982-10-01',f'1995-09-30', freq='d')

forcing_start_id = dHBV_time_range_all.get_loc(test_span[0])
forcing_end_id = dHBV_time_range_all.get_loc(test_span[-1])+1

simulation = np.full((len(area_info),len(test_span)),np.nan)


for gageidx, gagei in enumerate(area_info):

    print(gagei, "has merit of ", len(area_info[gagei]['unitarea']))
    print(gagei, "area is ", area_river[gageidx], "maximum area is ", np.max(area_info[gagei]['uparea']))

routparaAll= {}

for gageidx, gagei in enumerate(area_info):
    gagei = str(gagei).zfill(8)
    key_info = [gagei]
    

    ods_gage = obs_mm_day[gageidx:gageidx+1,forcing_start_id:forcing_end_id,:]
    nan_percentage = np.isnan(ods_gage).sum() / ods_gage.size 

    obs_gage_root = zarr.open_group(zarr_save_path+'/'+gagei, mode = 'r')
    COMIDs = obs_gage_root['COMID'][:]

    print(gagei, "has merit of ", len(COMIDs))

    unitarea = area_info[gagei]['unitarea']
    area_all = np.sum(unitarea)
    unitarea_fraction = unitarea/area_all

    unitarea_fraction_reshaped = unitarea_fraction[:, np.newaxis]

    xTrain2 = np.full((len(COMIDs),len(test_span),len(forcing_list_water_loss)),np.nan)
    attr2 = np.full((len(COMIDs),len(attributeLst_water_loss)),np.nan)


    for fid, foring_ in enumerate(forcing_list_water_loss):    
        xTrain2[:,:,fid] = obs_gage_root[foring_][:,forcing_start_id:forcing_end_id]
    # load attributes

    for aid, attribute_ in enumerate(attributeLst_water_loss) :                                    
        attr2[:,aid] =  obs_gage_root['attr'][attribute_][:] 



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

    #out = out +"/temporal_test/"
    if os.path.exists(out) is False:
        os.mkdir(out)
    testbatch =2000 #len(indexes)

    filePathLst = [out+"/Qs.csv",out+"/Q0.csv",out+"/Q1.csv",out+"/Q2.csv",out+"/ET.csv",out+"/SWE.csv",out+"/Qpp.csv"]



    testmodel.inittime = 0

    routpara = train.testModel_large_river(
        testmodel, testTuple, [forcing_norm2,attribute_norm2], c=None, area_info = area_info,gage_key = key_info,  batchSize=testbatch, filePathLst=filePathLst)
    routscaLst = [[0,2.9], [0,6.5]] 
    routpara[:,0] = routscaLst[0][0] + (routscaLst[0][1]-routscaLst[0][0])*routpara[:,0]
    routpara[:,1] = routscaLst[1][0] + (routscaLst[1][1]-routscaLst[1][0])*routpara[:,1]
    routpara_gage = np.sum(routpara*unitarea_fraction_reshaped, axis = 0,keepdims=True)
    
    routparaAll[gagei] = routpara_gage[0,:].tolist()


    #     dataPred = pd.read_csv(  out+"/Qs.csv", dtype=np.float32, header=None).values

    #     simulation[gageidx,:] = dataPred


    #     dataPred = np.expand_dims(dataPred, axis=-1)


    #     evaDict = [stat.statError(dataPred[:,len(warmup_span):,0], ods_gage[:,len(warmup_span):,0])]
    #     evaDictLst = evaDict
    #     keyLst = ['NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
    #     dataBox = list()
    #     for iS in range(len(keyLst)):
    #         statStr = keyLst[iS]
    #         temp = list()
    #         for k in range(len(evaDictLst)):
    #             data = evaDictLst[k][statStr]
    #             #data = data[~np.isnan(data)]
    #             temp.append(data)
    #         dataBox.append(temp)


    #     print(f"dHBV model at gage {gagei} 'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
    #         np.nanmedian(dataBox[0][0]),
    #         np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
    #         np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))






    #     variables_name = ['Qs', 'Q0', 'Q1', 'Q2','ET','SWE','Qpp']

    #     data_arrays = {}
    #     for idx, var_x in enumerate(variables_name):
    #         dataPred = pd.read_csv(  out+f"/{var_x}.csv", dtype=np.float32, header=None).values
    #         dataPred = np.expand_dims(dataPred, axis=-1)
    #         data_array = xr.DataArray(
    #             dataPred[:,:,0],
    #             dims = ['GAGE','time'],
    #             coords = {'GAGE':[gagei],
    #                         'time':test_span}
    #         )

    #         data_arrays[var_x] = data_array


    #     data_array = xr.DataArray(
    #         ods_gage[:,:,0],
    #         dims = ['GAGE','time'],
    #         coords = {'GAGE':[gagei],
    #                     'time':test_span}
    #     )

    #     data_arrays['runoff'] = data_array


    #     data_array = xr.DataArray(
    #         obs_mm_day[gageidx:gageidx+1,forcing_start_id:forcing_end_id,0],
    #         dims = ['GAGE','time'],
    #         coords = {'GAGE':[gagei],
    #                     'time':test_span}
    #     )

    #     data_arrays['streamflow'] = data_array

    #     xr_dataset = xr.Dataset(data_arrays)
    #     xr_dataset.to_zarr(store=out, group=f'largeRiver_simulation/{gagei}', mode='w')

    #     plt.figure(figsize=(12, 6))
    #     plt.plot(test_span[len(warmup_span):],dataPred[0,len(warmup_span):,0], label='dHBV', color='green')
    #   #  plt.plot(test_span[len(warmup_span):], NWM_runoff[0,:,0], label='NWM', color='red')
    #     plt.plot(test_span[len(warmup_span):], ods_gage[0,len(warmup_span):,0], label='Obs', color='blue')
    #     plt.title(f'Time Series of USGS__{gagei}, NSE of {round(dataBox[0][0][0],3)}, area of {area_river[gageidx]}')
    #     plt.xlabel('Date')
    #     plt.ylabel('Q (mm/day)')
    #     plt.legend()
    #     plt.minorticks_on()
    #     plt.grid(which='both', linestyle='--', linewidth=0.5)
    #     plt.grid(which='minor', linestyle=':', linewidth=0.5)
    #     plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/USGS_Ts/'+ f"USGS__{gagei}.png", dpi=300)


with open(out+'/large_river_simulations_rout_para.json','w') as f: 
    json.dump(routparaAll, f)

# ods_gage_all = obs_mm_day[:,forcing_start_id:forcing_end_id,:]

# evaDict = [stat.statError(simulation[:,len(warmup_span):], ods_gage_all[:,len(warmup_span):,0])]
# evaDictLst = evaDict
# keyLst = ['NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
# dataBox = list()
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     temp = list()
#     for k in range(len(evaDictLst)):
#         data = evaDictLst[k][statStr]
#         #data = data[~np.isnan(data)]
#         temp.append(data)
#     dataBox.append(temp)


# print(f"dHBV model at all gage 'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
#     np.nanmedian(dataBox[0][0]),
#     np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
#     np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))


# np.save(out+"large_river_simulations.npy",simulation)