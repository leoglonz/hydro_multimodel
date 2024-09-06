import numpy as np
import pandas as pd
import json
import xarray as xr
import zarr
import glob
import os

## Path to save the results
results_savepath = "/projects/mhpi/yxs275/DM_output/" + '/dPL_local_daymet_new_attr_water_loss_v6v10_random_batch_all_merit_forward/'



with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_info.json') as f:
    area_info = json.load(f)

GageID = area_info.keys()

COMID_all = []
for gage in GageID:
    COMIDs = area_info[gage]['COMID']
    COMID_all.extend(COMIDs)
    

COMID_sorted_unitque = list(set(COMID_all))
COMID_sorted_unitque.sort()


## Pick the regions/groups for forwarding

zonefileLst = glob.glob(results_savepath+"*")

zonefileLst.sort()

startyear = 1980
endyear = 2020
all_time_range = pd.date_range(start=f'{startyear}-01-01', end=f'{endyear}-12-31', freq='D')
## Pick the regions/groups for forwarding
time_range = pd.date_range(start=f'1980-10-01', end=f'2010-09-30', freq='D') 

start_idx = all_time_range.get_loc(time_range[0])
end_idx = all_time_range.get_loc(time_range[-1])+1

COMID_forward=[]
iter = 0
for id, zarr_file in enumerate(zonefileLst):

    root = zarr.open_group(zarr_file, mode = 'r')
    

    COMID_inzone = root['COMID'][:]
    COMID_inzone = [str(int(x)) for x in COMID_inzone]

    [C, ind1, SubInd] = np.intersect1d(COMID_sorted_unitque, COMID_inzone , return_indices=True)
    if ind1.any():
        COMID_forward.extend(list(np.array(COMID_sorted_unitque)[ind1])) 
        iter = iter+1
        if iter == 1 :
            collected_simulation = root['Qr'][SubInd,start_idx:end_idx]
        else:
            collected_simulation = np.concatenate((collected_simulation, root['Qr'][SubInd,start_idx:end_idx]), axis = 0)

        




selected_gage = []
Q_selected_gage = []
gageI = 0
for gage in GageID:
    
    unitarea = area_info[gage]['unitarea']
    uparea = area_info[gage]['uparea']
    COMIDs = area_info[gage]['COMID']

    [C, ind1, SubInd] = np.intersect1d(COMIDs, COMID_forward , return_indices=True)
    # if (len(ind1) == len(COMIDs)):
    #if abs(len(ind1) - len(COMIDs))/len(COMIDs)<0.1:
    gageI = gageI +1
    # if np.max(uparea)>5000:
        # print(np.max(uparea))
    print(gageI, "Gage ", gage,'is selected')
    selected_gage.append(gage)
    unitarea_in = np.array(unitarea)[ind1]
    temparea = np.tile(np.expand_dims(unitarea_in,axis = -1), (1, len(time_range)))

    Q_pred_gage = np.sum(collected_simulation[np.array(SubInd),:]*temparea, axis = 0)/np.sum(np.array(unitarea_in))
    Q_selected_gage.append(np.expand_dims(Q_pred_gage,axis = 0))

Q_selected = np.concatenate(Q_selected_gage,axis = 0)


data_array = xr.DataArray(
    Q_selected,
    dims = ['gage','time'],
    coords = {'gage':selected_gage,
            'time':time_range}
)

simulation_ds = xr.Dataset({'simulation_data': data_array})

# Saving the Dataset to Zarr format
simulation_ds.to_zarr(store=results_savepath, group='CONUS_gage_all',mode='w')


print("Done")