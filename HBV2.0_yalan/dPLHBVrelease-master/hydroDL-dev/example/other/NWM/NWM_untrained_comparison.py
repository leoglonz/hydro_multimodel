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
from scipy.spatial import KDTree
import glob
from hydroDL.post import  stat



from hydroDL.data import scale
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from hydroDL.data import scale


SWE_crd = pd.read_csv("/projects/mhpi/data/NWM/snow/SNOTEL_filter_data_1988/crd/crd.csv")
lat_snotel = SWE_crd['lat'].values
lon_snotel = SWE_crd['lon'].values
snotel_time_range = pd.date_range(start=f'{2001}-01-01', end=f'{2019}-12-31', freq='D')

snotel_SWE_path = '/projects/mhpi/data/NWM/snow/SNOTEL_filter_data_1988/'

snotel_SWE = np.full((len(lat_snotel),len(snotel_time_range)),np.nan)

for year in range(2001, 2020):
    SWE_year = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    start_id = snotel_time_range.get_loc(SWE_year[0])
    end_id = snotel_time_range.get_loc(SWE_year[-1])+1
    SWE_data = pd.read_csv(snotel_SWE_path+f'{year}/'+'SWE.csv',skiprows = 0,header=None)
    snotel_SWE[:,start_id:end_id] = SWE_data*1000


NWM_simulation_path = '/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/daily_SWE_updated/'
NWM_time_range = pd.date_range(start=f'{1979}-02-01', end=f'{2023}-02-01', freq='D')

NWM_SWE = np.full((len(lat_snotel),len(snotel_time_range)),np.nan)
NWM_start_id = NWM_time_range.get_loc(snotel_time_range[0])
NWM_end_id = NWM_time_range.get_loc(snotel_time_range[-1])+1

for idx in range(len(lat_snotel)):
    zarr_file = NWM_simulation_path+f'{lat_snotel[idx]}_{lon_snotel[idx]}'

    NWM_SWE_root = zarr.open_group(zarr_file, mode = 'r')
    NWM_SWE[idx,:] = NWM_SWE_root['SWE_NWM'][NWM_start_id:NWM_end_id]



evaDict = [stat.statError(NWM_SWE, snotel_SWE)]
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


print("NWM model'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax'",
      np.nanmedian(dataBox_NWM[0][0]),
      np.nanmedian(dataBox_NWM[1][0]), np.nanmedian(dataBox_NWM[2][0]), np.nanmedian(dataBox_NWM[3][0]),
      np.nanmedian(dataBox_NWM[4][0]), np.nanmedian(dataBox_NWM[5][0]))


dHBV_simulation_path = "/projects/mhpi/yxs275/DM_output/" + '/dPL_local_daymet_new_attr_water_loss_v6v14_random_batch_all_merit_forward/'

AORC_forcing_data_folder = '/projects/mhpi/hjj5218/data/NWM/distributed_HBV/'

forcing_data_folder = '/projects/mhpi/data/conus/zarr/'

dHBV_time_range = pd.date_range(start=f'{1980}-01-01', end=f'{2020}-12-31', freq='D')

AORC_precip = np.full((len(lat_snotel),len(snotel_time_range)),np.nan)

dHBV_SWE = np.full((len(lat_snotel),len(snotel_time_range)),np.nan)
dHBV_precip = np.full((len(lat_snotel),len(snotel_time_range)),np.nan)
dHBV_start_id = dHBV_time_range.get_loc(snotel_time_range[0])
dHBV_end_id = dHBV_time_range.get_loc(snotel_time_range[-1])+1

AORC_T = np.full((len(lat_snotel),len(snotel_time_range)),np.nan)
dHBV_T = np.full((len(lat_snotel),len(snotel_time_range)),np.nan)

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

# Create KDTree for efficient spatial matching
merit_coords = list(zip(MERIT_lat, MERIT_lon))
snotel_coords = list(zip(lat_snotel, lon_snotel))

kd_tree = KDTree(merit_coords)

# Find the nearest MERIT point for each SNOTEL point
distances, indices = kd_tree.query(snotel_coords)

MERIT_COMID_reordered = [MERIT_COMID[idx] for idx in indices]
MERIT_lat_reordered = [MERIT_lat[idx] for idx in indices]
MERIT_lon_reordered = [MERIT_lon[idx] for idx in indices]

var_c_list = ['aridity', 'meanP', 'ETPOT_Hargr', 'NDVI', 'FW', 'meanslope', 'SoilGrids1km_sand', 'SoilGrids1km_clay',
           'SoilGrids1km_silt', 'glaciers', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt',
           'meanelevation', 'meanTa', 'permafrost', 'permeability',
           'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction','T_clay','T_gravel','T_sand','T_silt','Porosity','catchsize']


subzonefile_lst = []
zonefileLst = glob.glob(dHBV_simulation_path+"*")

zonelst =[(x.split("/")[-1]) for x in zonefileLst]
zonelst.sort()
forwarded_COMID = np.full((len(lat_snotel)),np.nan)
attr = np.full((len(lat_snotel),len(var_c_list)),np.nan)
for idx in range(len(zonelst)):

    print("Working on zone ", zonelst[idx])
    root_zone = zarr.open_group(dHBV_simulation_path+zonelst[idx], mode = 'r')
    forcing_root_zone = zarr.open_group(forcing_data_folder+zonelst[idx][:2]+"/"+zonelst[idx], mode = 'r')
    AORC_forcing_root_zone = zarr.open_group(AORC_forcing_data_folder+zonelst[idx][:2]+"/"+zonelst[idx], mode = 'r')
    gage_COMIDs = root_zone['COMID'][:]

    [C, ind1, SubInd] = np.intersect1d(MERIT_COMID_reordered, gage_COMIDs, return_indices=True)
    
    forwarded_COMID[ind1] = np.array(MERIT_COMID_reordered)[ind1]
    if SubInd.any():

        dHBV_SWE[ind1,:] = root_zone['SWE'][SubInd,dHBV_start_id:dHBV_end_id]
        dHBV_precip[ind1,:] = forcing_root_zone['P'][SubInd,dHBV_start_id:dHBV_end_id]
        AORC_precip[ind1,:] = AORC_forcing_root_zone['P'][SubInd,NWM_start_id:NWM_end_id]
        dHBV_T[ind1,:] = forcing_root_zone['Temp'][SubInd,dHBV_start_id:dHBV_end_id]
        AORC_T[ind1,:] = AORC_forcing_root_zone['Temp'][SubInd,NWM_start_id:NWM_end_id]
        for variablei, variable in enumerate(var_c_list):
            attr[ind1,variablei]= forcing_root_zone['attrs'][variable][SubInd]


new_sim = "/projects/mhpi/data/NWM/snow/snotel_point"
new_root_zone = zarr.open_group(new_sim, mode = 'r')

new_SWE_time_range = pd.date_range(start=f'{1987}-10-01', end=f'{2020}-12-31', freq='D')
new_SWE = new_root_zone['SWE'][:,new_SWE_time_range.get_loc(snotel_time_range[0]):new_SWE_time_range.get_loc(snotel_time_range[-1])+1]
attr_df = pd.DataFrame(attr,  columns=var_c_list)

attr_df.to_csv( "/projects/mhpi/data/NWM/snow/attr.csv", index=True)

with open('/projects/mhpi/yxs275/tools/extrect_merit_in_basin/'+'area_all_merit_info.json') as f:
    area_all_merit_info = json.load(f)

Ac = [area_all_merit_info[x]['uparea'][0] for x in MERIT_COMID_reordered]
Ai = [area_all_merit_info[x]['unitarea'][0] for x in MERIT_COMID_reordered]

Snotel_attr = pd.read_csv("/projects/mhpi/data/NWM/snow/SNOTEL_filter_data_1988/attributes_all.csv")
elevation =  Snotel_attr['mean_elev'].values

# selected_low_elevation_idx = np.where(elevation>3000)[0]

evaDict = [stat.statError(new_SWE, snotel_SWE)]
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


print("dHBV model'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','rdMax'",
      np.nanmedian(dataBox_dHBV[0][0]),
      np.nanmedian(dataBox_dHBV[1][0]), np.nanmedian(dataBox_dHBV[2][0]), np.nanmedian(dataBox_dHBV[3][0]),
      np.nanmedian(dataBox_dHBV[4][0]), np.nanmedian(dataBox_dHBV[5][0]))

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
tmean = (site_daymet_tmin+site_daymet_tmax)/2




df = pd.DataFrame(site_daymet_forcing.T, index=snotel_time_range, columns=range(len(lat_snotel)))

# Define a function to determine the water year
def water_year(date):
    year = date.year
    if date.month >= 10:
        return year + 1
    else:
        return year

# Apply the function to create a water year column
df['water_year'] = df.index.map(water_year)

# Initialize an empty DataFrame to store cumulative precipitation
AORC_cumulative_df = pd.DataFrame(index=df.index)

# Calculate cumulative precipitation for each site
for site in df.columns[:-1]:  # Exclude the 'water_year' column
    site_df = df[[site, 'water_year']].copy()
    site_df['cumulative_precip'] = site_df.groupby('water_year')[site].cumsum()
    AORC_cumulative_df[site] = site_df['cumulative_precip']





df = pd.DataFrame(AORC_precip.T, index=snotel_time_range, columns=range(len(lat_snotel)))

# Define a function to determine the water year
def water_year(date):
    year = date.year
    if date.month >= 10:
        return year + 1
    else:
        return year

# Apply the function to create a water year column
df['water_year'] = df.index.map(water_year)

# Create a DataFrame for temperature (assuming T is defined and has the same dimensions as AORC_precip)
temperature_df = pd.DataFrame(tmean.T, index=snotel_time_range, columns=range(len(lat_snotel)))

# Initialize an empty DataFrame to store cumulative precipitation
AORC_cumulative_df = pd.DataFrame(index=df.index)

# Calculate cumulative precipitation for each site, considering temperature
for site in df.columns[:-1]:  # Exclude the 'water_year' column
    site_df = df[[site, 'water_year']].copy()
    site_df['temperature'] = temperature_df[site]
    
    # Only accumulate precipitation when temperature > 0
    site_df['precip_accum'] = site_df[site].where(site_df['temperature'] < 30, 0)
    
    # Calculate cumulative precipitation by water year
    site_df['cumulative_precip'] = site_df.groupby('water_year')['precip_accum'].cumsum()
    
    AORC_cumulative_df[site] = site_df['cumulative_precip']

# Remove the temporary 'temperature' and 'precip_accum' columns if needed
AORC_cumulative_df = AORC_cumulative_df.drop(columns=['temperature', 'precip_accum'], errors='ignore')







i = 250
plt.figure(figsize=(12, 6))
plt.plot(snotel_time_range, NWM_SWE[i, :]/25.4, label='NWM', color='red')
plt.plot(snotel_time_range, new_SWE[i, :]/25.4, label='dHBV', color='blue')
plt.plot(snotel_time_range, snotel_SWE[i, :]/25.4, label='SNOTEL', color='orange')
# plt.plot(snotel_time_range, tmean[i, :], label='Tmep', color='purple')
# plt.plot(snotel_time_range, site_daymet_cumulative_df.values.T[i, :], label='Daymet site Precip', color='purple')
# plt.plot(snotel_time_range, dHBV_cumulative_df.values.T[i, :], label='Daymet basin Precip', color='green')
plt.plot(snotel_time_range, AORC_cumulative_df.values.T[i, :]/25.4, label='Accumulated Precip', color='k')
plt.title(f'Time Series of {np.array(MERIT_COMID_reordered)[i]} ')
plt.xlabel('Date')
plt.ylabel('SWE (inches)')
plt.legend()
plt.minorticks_on()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/'+ f"SWE_Ts.png", dpi=300)
