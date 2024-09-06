import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))

import zarr
import numpy as np
import pandas as pd
import json
import xarray as xr
import os
import glob
import torch
import torch.nn.functional as F
from hydroDL.post import  stat




import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from hydroDL.data import scale


def UH_conv(x,UH,viewmode=1):
    # UH is a vector indicating the unit hydrograph
    # the convolved dimension will be the last dimension
    # UH convolution is
    # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    # hence we flip the UH
    # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    # view
    # x: [batch, var, time]
    # UH:[batch, var, uhLen]
    # batch needs to be accommodated by channels and we make use of groups
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/nn.functional.html

    mm= x.shape; nb=mm[0]
    m = UH.shape[-1]
    padd = m-1
    if viewmode==1:
        xx = x.view([1,nb,mm[-1]])
        w  = UH.view([nb,1,m])
        groups = nb

    y = F.conv1d(xx, torch.flip(w,[2]), groups=groups, padding=padd, stride=1, bias=None)
    y=y[:,:,0:-padd]
    return y.view(mm)


def UH_gamma(a,b,lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    w = torch.zeros([lenF, m[1],m[2]])
    aa = F.relu(a[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.1 # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.5 # minimum 0.5
    t = torch.arange(0.5,lenF*1.0).view([lenF,1,1]).repeat([1,m[1],m[2]])
    t = t.cuda(aa.device)
    denom = (aa.lgamma().exp())*(theta**aa)
    mid= t**(aa-1)
    right=torch.exp(-t/theta)
    w = 1/denom*mid*right
    w = w/w.sum(0) # scale to 1 for each UH

    return w

traingpuid = 0
torch.cuda.set_device(traingpuid)

main_river_info_Path = '/projects/mhpi/hjj5218/data/Main_River/CONUS_v3/'
dHBV_simulation_path_1 = "/projects/mhpi/yxs275/DM_output/" + '/dPL_local_daymet_new_attr_water_loss_v6v14_random_batch_all_merit_forward/'

dHBV_simulation_path_2 = "/projects/mhpi/yxs275/DM_output/" + '/dPL_local_daymet_new_attr_water_loss_v6v18_random_batch_filled_data_dynamic_K0_correct_nfea2/'

dHBV_time_range_all = pd.date_range(start=f'{1980}-01-01', end=f'{2020}-12-31', freq='D')
dHBV_time_range = pd.date_range(start=f'{1981}-10-01', end=f'{2020}-09-30', freq='D')
dHBV_start_id = dHBV_time_range_all.get_loc(dHBV_time_range[0])
dHBV_end_id = dHBV_time_range_all.get_loc(dHBV_time_range[-1])+1
Nstep = len(dHBV_time_range)

subzonefile_lst = []
zonefileLst = glob.glob(dHBV_simulation_path_1+"*")

zonelst =[(x.split("/")[-1]) for x in zonefileLst]
zonelst.sort()



with open(main_river_info_Path + 'area_info_main_river.json') as f:
    main_river_info = json.load(f)


rootOut_1 = "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v14_random_batch_filled_data_dynamic_K0/'
out_1 = os.path.join(rootOut_1, "exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test/")  # output folder to save results

with open(out_1 + 'large_river_simulations_rout_para.json') as f:
    rout_para_info_1 = json.load(f)

rootOut_2 = "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v18_random_batch_filled_data_dynamic_K0_correct_nfea2/'
out_2 = os.path.join(rootOut_2, "exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test/")  # output folder to save results

with open(out_2 + 'large_river_simulations_rout_para.json') as f:
    rout_para_info_2 = json.load(f)

# attribut_river_file = "/projects/mhpi/hjj5218/data/Main_River/CONUS_v2/main_river_41.csv"
# attribut_river = pd.read_csv(attribut_river_file)
# area_river = attribut_river['area'].values

# obs_ft = np.load('/projects/mhpi/hjj5218/data/Main_River/CONUS_v2/obs_streamflow_1980_2020.npy')/0.0283168 

# obs_ft = np.swapaxes(obs_ft,1,0)
# obs_mm_day = scale._basin_norm(
#                         np.expand_dims(obs_ft,axis = -1 ) ,  np.expand_dims(area_river,axis = -1), to_norm=True
#                     )  ## from ft^3/s to mm/day





with open(out_1+'/large_river_simulations_rout_para.json') as f: 
    routparaAll_1 = json.load(f)

with open(out_2+'/large_river_simulations_rout_para.json') as f: 
    routparaAll_2 = json.load(f)

large_river_file = glob.glob("/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/large_river_daily_simulation/v2/"+"*")

River_id =[(x.split("/")[-1]) for x in large_river_file]
River_id.sort()

obs_path = '/projects/mhpi/hjj5218/data/Main_River/CONUS_v3/distributed.zarr/'


obs_test = np.full((len(River_id),len(dHBV_time_range)),np.nan)
dHBV_simulation_1 = np.full((len(River_id),len(dHBV_time_range)),np.nan)
dHBV_simulation_2 = np.full((len(River_id),len(dHBV_time_range)),np.nan)
NWM_simulation_all = np.full((len(River_id),len(dHBV_time_range)),np.nan)
# bad_rivers =[
#     "02427500","02429500",'02470500', "03159870", "03160000", 
#     "06879000", "06888345", "07138062", "07138065", 
#     "07355500","08092600", "08447300", "09514300", "09158500",'09479501', 
#     '09514300','09518500',"09520280", "09520700", "12391000","13171620","13290200"
# ]
basin_area_all = np.full((len(River_id),1),np.nan)



LSTM_rootOut = "/projects/mhpi/yxs275/model/"+'LSTM_local_daymet_filled_withNaN_NSE_with_same_forcing_HBV_2800/'
LSTM_out = os.path.join(LSTM_rootOut, "exp_EPOCH300_BS100_RHO365_HS512_trainBuff365/")  # output folder to save results


LSTM_root =  zarr.open_group(LSTM_out+'large_river.zarr', mode = 'r')
LSTM_simulation = LSTM_root['large_river']['LSTM_Qs']



for i in range(len(River_id)):
    
    COMIDs = main_river_info[River_id[i]]['COMID']
    unitarea = main_river_info[River_id[i]]['unitarea']
    uparea = main_river_info[River_id[i]]['uparea']
    area_all = np.sum(unitarea)
    unitarea_fraction = unitarea/area_all

    unitarea_fraction_reshaped = unitarea_fraction[:, np.newaxis]

    dHBV_Q_merit_1 = np.full((len(COMIDs),len(dHBV_time_range)),np.nan)
    dHBV_Q_merit_2 = np.full((len(COMIDs),len(dHBV_time_range)),np.nan)
   

    idx_all = []
    for idx in range(len(zonelst)):

        print("Working on zone ", zonelst[idx])
        root_zone_1 = zarr.open_group(dHBV_simulation_path_1+zonelst[idx], mode = 'r')
        root_zone_2 = zarr.open_group(dHBV_simulation_path_2+zonelst[idx], mode = 'r')

        gage_COMIDs = root_zone_1['COMID'][:]

        [C, ind1, SubInd] = np.intersect1d(COMIDs, gage_COMIDs, return_indices=True)
        idx_all.extend(ind1)
        if SubInd.any():

            dHBV_Q_merit_1[ind1,:] = root_zone_1['Q_not_routed'][SubInd,dHBV_start_id:dHBV_end_id]
            dHBV_Q_merit_2[ind1,:] = root_zone_2['Q_not_routed'][SubInd,dHBV_start_id:dHBV_end_id]

    print((len(COMIDs)-len(idx_all))/len(COMIDs)*100,"% of merit does not have data")
    dHBV_Q_1 = np.nansum(dHBV_Q_merit_1*unitarea_fraction_reshaped, axis = 0,keepdims=True)
    dHBV_Q_2 = np.nansum(dHBV_Q_merit_2*unitarea_fraction_reshaped, axis = 0,keepdims=True)

    

    dHBV_Q_1 = torch.from_numpy(dHBV_Q_1).cuda().permute([1, 0])
    dHBV_Q_2 = torch.from_numpy(dHBV_Q_2).cuda().permute([1, 0])

    routa_1 =torch.from_numpy(np.array(rout_para_info_1[River_id[i]][0])).cuda()
    routb_1 = torch.from_numpy(np.array(rout_para_info_1[River_id[i]][1])).cuda()

    routa_1 = routa_1.repeat(Nstep, 1).unsqueeze(-1)
    routb_1 = routb_1.repeat(Nstep, 1).unsqueeze(-1)
    UH_1 = UH_gamma(routa_1, routb_1, lenF=15)  # lenF: folter
    rf_1 = torch.unsqueeze(dHBV_Q_1, -1).permute([1, 2, 0])   # dim:gage*var*time
    UH_1 = UH_1.permute([1, 2, 0])  # dim: gage*var*time
    Qsrout_1= UH_conv(rf_1, UH_1).permute([2, 0, 1])


    Qs_1 = Qsrout_1.squeeze(-1).detach().cpu().numpy().swapaxes(0, 1)


    routa_2 =torch.from_numpy(np.array(rout_para_info_2[River_id[i]][0])).cuda()
    routb_2 = torch.from_numpy(np.array(rout_para_info_2[River_id[i]][1])).cuda()

    routa_2 = routa_2.repeat(Nstep, 1).unsqueeze(-1)
    routb_2 = routb_2.repeat(Nstep, 1).unsqueeze(-1)
    UH_2 = UH_gamma(routa_2, routb_2, lenF=15)  # lenF: folter
    rf_2 = torch.unsqueeze(dHBV_Q_2, -1).permute([1, 2, 0])   # dim:gage*var*time
    UH_2 = UH_2.permute([1, 2, 0])  # dim: gage*var*time
    Qsrout_2= UH_conv(rf_2, UH_2).permute([2, 0, 1])


    Qs_2 = Qsrout_2.squeeze(-1).detach().cpu().numpy().swapaxes(0, 1)








    root_obs = zarr.open_group(obs_path+River_id[i], mode = 'r')
    obs_i = root_obs['streamflow'][dHBV_start_id:dHBV_end_id]/0.0283168 
    basin_area = root_obs['DRAIN_SQKM'][:]

    basin_area_all[i] = basin_area
    obs_mm_day = scale._basin_norm(
                        np.expand_dims(np.expand_dims(obs_i,axis = -1 ),axis = 0 ),  np.expand_dims(basin_area,axis = -1), to_norm=True
                    )  ## from ft^3/s to mm/day

    obs_test[i,:] = obs_mm_day[0,:,0]
    dHBV_simulation_1[i,:] =Qs_1[0,:]
    dHBV_simulation_2[i,:] =Qs_2[0,:]
    evaDict = [stat.statError(Qs_1, obs_mm_day[:,:,0])]
    evaDictLst = evaDict
    keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax']
    dataBox_dHBV1 = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(evaDictLst)):
            data = evaDictLst[k][statStr]
            #data = data[~np.isnan(data)]
            temp.append(data)
        dataBox_dHBV1.append(temp)


    print(f"dHBV large river v6.14 {River_id[i]}'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax'",
        np.nanmedian(dataBox_dHBV1[0][0]),
        np.nanmedian(dataBox_dHBV1[1][0]), np.nanmedian(dataBox_dHBV1[2][0]), np.nanmedian(dataBox_dHBV1[3][0]),
        np.nanmedian(dataBox_dHBV1[4][0]), np.nanmedian(dataBox_dHBV1[5][0]))



    evaDict = [stat.statError(Qs_2, obs_mm_day[:,:,0])]
    evaDictLst = evaDict
    keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax']
    dataBox_dHBV2 = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(evaDictLst)):
            data = evaDictLst[k][statStr]
            #data = data[~np.isnan(data)]
            temp.append(data)
        dataBox_dHBV2.append(temp)


    print(f"dHBV large river v6.18 {River_id[i]}'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax'",
        np.nanmedian(dataBox_dHBV2[0][0]),
        np.nanmedian(dataBox_dHBV2[1][0]), np.nanmedian(dataBox_dHBV2[2][0]), np.nanmedian(dataBox_dHBV2[3][0]),
        np.nanmedian(dataBox_dHBV2[4][0]), np.nanmedian(dataBox_dHBV2[5][0]))




    NWM_path = "/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/large_river_daily_simulation/v2/"
    NWM_root = zarr.open_group(NWM_path+River_id[i], mode = 'r')
    NWM_timespan = pd.date_range('1979-02-01',f'2023-02-01', freq='d')

    NWM_start = NWM_timespan.get_loc(dHBV_time_range[0])
    NWM_end = NWM_timespan.get_loc(dHBV_time_range[-1])+1
    feature_id = NWM_root['feature_id'][:]
    for ii in range(len(feature_id )):
        NWM_simulation = NWM_root['Qs_NWM'][ii:ii+1,NWM_start:NWM_end]/0.0283168 


        NWM_runoff = scale._basin_norm(
                                np.expand_dims(NWM_simulation,axis = -1 ) ,  np.expand_dims(basin_area,axis = -1), to_norm=True
                            )  ## from ft^3/s to mm/day

        evaDict = [stat.statError(NWM_runoff[:,:,0], obs_mm_day[:,:,0])]
        evaDictLst = evaDict
        keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax']
        dataBox_NWM = list()
        for iS in range(len(keyLst)):
            statStr = keyLst[iS]
            temp = list()
            for k in range(len(evaDictLst)):
                data = evaDictLst[k][statStr]
                #data = data[~np.isnan(data)]
                temp.append(data)
            dataBox_NWM.append(temp)


        print(f"NWM large rive {River_id[i]} at feature id {feature_id[ii]}'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax'",
            np.nanmedian(dataBox_NWM[0][0]),
            np.nanmedian(dataBox_NWM[1][0]), np.nanmedian(dataBox_NWM[2][0]), np.nanmedian(dataBox_NWM[3][0]),
            np.nanmedian(dataBox_NWM[4][0]), np.nanmedian(dataBox_NWM[5][0]))


    NWM_simulation_all[i,:] =NWM_runoff[0,:,0]


    LSTM_simulation_i = LSTM_simulation[i:i+1,:]

    evaDict = [stat.statError(LSTM_simulation_i, obs_mm_day[:,:,0])]
    evaDictLst = evaDict
    keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax']
    dataBox_LSTM = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(evaDictLst)):
            data = evaDictLst[k][statStr]
            #data = data[~np.isnan(data)]
            temp.append(data)
        dataBox_LSTM.append(temp)


    print(f"LSTM large rive {River_id[i]}'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax'",
        np.nanmedian(dataBox_LSTM[0][0]),
        np.nanmedian(dataBox_LSTM[1][0]), np.nanmedian(dataBox_LSTM[2][0]), np.nanmedian(dataBox_LSTM[3][0]),
        np.nanmedian(dataBox_LSTM[4][0]), np.nanmedian(dataBox_LSTM[5][0]))




    plt.figure(figsize=(12, 6))

    plt.plot(dHBV_time_range, obs_mm_day[0,:,0], label='Obs', color='blue',marker = 'o', markerfacecolor='none')
    plt.plot(dHBV_time_range, Qs_1[0,:], label='dHBV', color='green')
    plt.plot(dHBV_time_range, Qs_2[0,:], label='dHBV', color='lightgreen')
    plt.plot(dHBV_time_range, NWM_runoff[0,:,0], label='NWM', color='red') 
    plt.plot(dHBV_time_range, LSTM_simulation_i[0,:], label='LSTM', color='pink')
    plt.title(f'Time Series of USGS_{River_id[i]} , NSE of dHBV2.0 v6.14 {round(dataBox_dHBV1[0][0][0],3)}, NSE of dHBV2.0 v6.18 {round(dataBox_dHBV2[0][0][0],3)}, \n  NSE of NWM {round(dataBox_NWM[0][0][0],3)},  NSE of LSTM {round(dataBox_LSTM[0][0][0],3)}, area of {basin_area}')
    plt.xlabel('Date')
    plt.ylabel('Q (mm/day)')
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/USGS_Ts/'+ f"large_river_Ts_{River_id[i]}_rout.png", dpi=300)





evaDict = [stat.statError(dHBV_simulation_1, obs_test)]
evaDictLst_dHBV1 = evaDict
keyLst = ['NSE', 'KGE','Bias','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst_dHBV1)):
        data = evaDictLst_dHBV1[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)


print("dHBV model v6.14'NSE', 'KGE','Bias','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmean(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]), np.nanmedian(dataBox[9][0]))





evaDict = [stat.statError(dHBV_simulation_2, obs_test)]
evaDictLst_dHBV2 = evaDict
keyLst = ['NSE', 'KGE','Bias','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst_dHBV2)):
        data = evaDictLst_dHBV2[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)


print("dHBV model v6.18'NSE', 'KGE','Bias','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmean(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]), np.nanmedian(dataBox[9][0]))





evaDict = [stat.statError(LSTM_simulation, obs_test)]
evaDictLst_LSTM = evaDict
keyLst = ['NSE', 'KGE','Bias','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst_LSTM)):
        data = evaDictLst_LSTM[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)


print("LSTM model'NSE', 'KGE','Bias','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmean(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]), np.nanmedian(dataBox[9][0]))








evaDict = [stat.statError(NWM_simulation_all, obs_test)]
evaDictLst_NWM = evaDict
keyLst = ['NSE', 'KGE','Bias','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst_NWM)):
        data = evaDictLst_NWM[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)


print("NWM model'NSE', 'KGE','Bias','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmean(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]), np.nanmedian(dataBox[9][0]))





model_names = ['dHBV2.0_v6_14', 'dHBV2.0_v6_18', 'LSTM','NWM3.0']

# Combining data for boxplot
combined_data = {'NSE': [], 'Bias': []}
for key in combined_data.keys():
    combined_data[key] = [evaDictLst_dHBV1[0][key], evaDictLst_dHBV2[0][key], evaDictLst_LSTM[0][key], evaDictLst_NWM[0][key]]

# Creating boxplots
plt.rcParams.update({'font.size': 22})

box_colors = ['pink', 'red', 'mediumpurple', 'blue']

fig, axs = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)
bplot0 = axs[0].boxplot(combined_data['NSE'], showfliers=False, labels=model_names,
                         patch_artist=True)

for patch, color in zip(bplot0['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('k')
    patch.set_linewidth(2)

for whisker in bplot0['whiskers']:
    whisker.set(ls='-', linewidth=2, color="black")
for cap in bplot0['caps']:
    cap.set(ls='-', linewidth=2, color="black")
for box in bplot0['boxes']:
    box.set(ls='-', linewidth=2)
for median in bplot0['medians']:
    median.set(ls='-', linewidth=2, color="black")

axs[0].set_title('NSE')
axs[0].set_ylabel('NSE Values')


bplot1= axs[1].boxplot(combined_data['Bias'],showfliers=False, labels=model_names,patch_artist=True)
for patch, color in zip(bplot1['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('k')
    patch.set_linewidth(2)

for whisker in bplot1['whiskers']:
    whisker.set(ls='-', linewidth=2, color="black")
for cap in bplot1['caps']:
    cap.set(ls='-', linewidth=2, color="black")
for box in bplot1['boxes']:
    box.set(ls='-', linewidth=2)
for median in bplot1['medians']:
    median.set(ls='-', linewidth=2, color="black")


axs[1].set_title('Bias')
axs[1].set_ylabel('Bias Values')

plt.suptitle('Model Evaluation Metrics')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/USGS_Ts/'+ f"metric_boxplot.png", dpi=300)











data_arrays = {}

   

data_array = xr.DataArray(
    dHBV_simulation_2,
    dims = ['COMID','time'],
    coords = {'COMID':River_id,
                'time':dHBV_time_range}
)

data_arrays['dHBV_Qs'] = data_array


data_array = xr.DataArray(
    obs_test,
    dims = ['COMID','time'],
    coords = {'COMID':River_id,
                'time':dHBV_time_range}
)

data_arrays['runoff'] = data_array




xr_dataset = xr.Dataset(data_arrays)
xr_dataset.to_zarr(store=out+"/large_river.zarr", group=f'large_river', mode='w')