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

main_river_info_Path = '/projects/mhpi/hjj5218/data/Main_River/CONUS/'
dHBV_simulation_path = "/projects/mhpi/yxs275/DM_output/" + '/dPL_local_daymet_new_attr_water_loss_v6v14_random_batch_all_merit_forward_cap_RT_for_Ac_larger_50000/'

dHBV_time_range_all = pd.date_range(start=f'{1980}-01-01', end=f'{2020}-12-31', freq='D')
dHBV_time_range = pd.date_range(start=f'{1985}-10-01', end=f'{1995}-09-30', freq='D')
dHBV_start_id = dHBV_time_range_all.get_loc(dHBV_time_range[0])
dHBV_end_id = dHBV_time_range_all.get_loc(dHBV_time_range[-1])+1
Nstep = len(dHBV_time_range)

subzonefile_lst = []
zonefileLst = glob.glob(dHBV_simulation_path+"*")

zonelst =[(x.split("/")[-1]) for x in zonefileLst]
zonelst.sort()

with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_all_merit_info.json') as f:
    area_all_merit_info = json.load(f)

with open(main_river_info_Path + 'area_info_main_river.json') as f:
    main_river_info = json.load(f)

River_id = list(main_river_info.keys())

i = 0
COMIDs = main_river_info[River_id[i]]['COMID']
unitarea = main_river_info[River_id[i]]['unitarea']
uparea = main_river_info[River_id[i]]['uparea']
area_all = np.sum(unitarea)
unitarea_fraction = unitarea/area_all

unitarea_fraction_reshaped = unitarea_fraction[:, np.newaxis]

dHBV_Q_merit = np.full((len(COMIDs),len(dHBV_time_range)),np.nan)

dHBV_RT_merit = np.full((len(COMIDs),len(dHBV_time_range)),np.nan)

idx_all = []
for idx in range(len(zonelst)):

    print("Working on zone ", zonelst[idx])
    root_zone = zarr.open_group(dHBV_simulation_path+zonelst[idx], mode = 'r')

    gage_COMIDs = root_zone['COMID'][:]

    [C, ind1, SubInd] = np.intersect1d(COMIDs, gage_COMIDs, return_indices=True)
    idx_all.extend(ind1)
    if SubInd.any():

        dHBV_Q_merit[ind1,:] = root_zone['Q_not_routed'][SubInd,dHBV_start_id:dHBV_end_id]
        dHBV_RT_merit[ind1,:] = root_zone['Q0'][SubInd,dHBV_start_id:dHBV_end_id]

print((len(COMIDs)-len(idx_all))/len(COMIDs)*100,"% of merit does not have data")
dHBV_Q = np.nansum(dHBV_Q_merit*unitarea_fraction_reshaped, axis = 0,keepdims=True)

dHBV_RT = np.nansum(dHBV_RT_merit*unitarea_fraction_reshaped, axis = 0,keepdims=True)

dHBV_Q = torch.from_numpy(dHBV_Q).cuda().permute([1, 0])

routa =torch.from_numpy(np.array([2.0909					])).cuda()
routb = torch.from_numpy(np.array([0.5613			])).cuda()

routa = routa.repeat(Nstep, 1).unsqueeze(-1)
routb = routb.repeat(Nstep, 1).unsqueeze(-1)
UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
rf = torch.unsqueeze(dHBV_Q, -1).permute([1, 2, 0])   # dim:gage*var*time
UH = UH.permute([1, 2, 0])  # dim: gage*var*time
Qsrout = UH_conv(rf, UH).permute([2, 0, 1])


Qs = Qsrout.squeeze(-1).detach().cpu().numpy().swapaxes(0, 1)

obs = np.load('/projects/mhpi/hjj5218/data/Main_River/CONUS/main_river_conus_1980_2020.npy')


obs_i = obs[i:i+1,dHBV_start_id:dHBV_end_id,0]




evaDict = [stat.statError(Qs, obs_i)]
evaDictLst = evaDict
keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax']
dataBox_dHBV = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox_dHBV.append(temp)


print(f"dHBV large rive {River_id[i]}'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax'",
      np.nanmedian(dataBox_dHBV[0][0]),
      np.nanmedian(dataBox_dHBV[1][0]), np.nanmedian(dataBox_dHBV[2][0]), np.nanmedian(dataBox_dHBV[3][0]),
      np.nanmedian(dataBox_dHBV[4][0]), np.nanmedian(dataBox_dHBV[5][0]))


NWM_path = "/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/large_river_daily_simulation/"
NWM_root = zarr.open_group(NWM_path+River_id[i], mode = 'r')
NWM_timespan = pd.date_range('1979-02-01',f'2023-02-01', freq='d')

NWM_start = NWM_timespan.get_loc(dHBV_time_range[0])
NWM_end = NWM_timespan.get_loc(dHBV_time_range[-1])+1
feature_id = NWM_root['feature_id'][:]
for ii in range(len(feature_id )):
    NWM_simulation = NWM_root['Qs_NWM'][ii:ii+1,NWM_start:NWM_end]/0.0283168 


    NWM_runoff = scale._basin_norm(
                            np.expand_dims(NWM_simulation,axis = -1 ) ,  np.expand_dims(area_all,axis = -1), to_norm=True
                        )  ## from ft^3/s to mm/day

    evaDict = [stat.statError(NWM_runoff[:,:,0], obs_i)]
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

plt.figure(figsize=(12, 6))
plt.plot(dHBV_time_range, Qs[0,:], label='dHBV', color='green')
plt.plot(dHBV_time_range, NWM_runoff[0,:,0], label='NWM', color='red')
plt.plot(dHBV_time_range, obs_i[0,:], label='Obs', color='blue')
plt.title(f'Time Series of {River_id[i]} ')
plt.xlabel('Date')
plt.ylabel('Q (mm/day)')
plt.legend()
plt.minorticks_on()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/'+ f"large_river_Ts_{River_id[i]}.png", dpi=300)
