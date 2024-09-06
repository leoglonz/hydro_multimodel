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

from hydroDL.post import  stat,plot



from hydroDL.data import scale
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from mpl_toolkits.basemap import Basemap
from hydroDL.data import scale

GAGE2_NWM = '/projects/mhpi/yxs275/Data/GAGES_2/NWM_data/'


NWM_cal_basin_path = "/projects/mhpi/hjj5218/data/NWM/zarr/"


NWM_cal_basin_files  = glob.glob(NWM_cal_basin_path+"/*")
NWM_cal_basin_gage =[x.split("/")[-1] for x in NWM_cal_basin_files]
NWM_cal_basin_gage.sort()


zarr_save_path = '/projects/mhpi/yxs275/Data/GAGES_2/NWM_data/'
obs_gage_root = zarr.open_group(zarr_save_path, mode = 'r')
NWM_data_gages = obs_gage_root['forcing']['gageID'][:]
NWM_data_gages = [str(x).zfill(8) for x in NWM_data_gages]


attribute_file = '/projects/mhpi/yxs275/Data/attributes_haoyu/attributes_haoyu.csv'
attributeALL_df = pd.read_csv(attribute_file,index_col=0)
attributeALL_df = attributeALL_df.sort_values(by='id')


gage_info_file_selected_from_merit = "/projects/mhpi/data/MERIT/gage_information/formatted_gage_csvs/gages_3000_merit_info.csv"
gage_info_from_merit = pd.read_csv(gage_info_file_selected_from_merit)

gage_info_from_merit = gage_info_from_merit.sort_values(by='STAID')
gageIDs_from_merit = gage_info_from_merit['STAID'].values

attributeALL_df  = attributeALL_df[attributeALL_df['id'].isin(gageIDs_from_merit)]

attributeALL_df = attributeALL_df[attributeALL_df['area'] > 500]

attributeAllLst = attributeALL_df.columns
training_dHBV_gages = attributeALL_df["id"].values

training_dHBV_gages = [str(x).zfill(8) for x in training_dHBV_gages]


idx_train = [idx for idx,id in enumerate(NWM_data_gages) if id in training_dHBV_gages]

idx_cal = [idx for idx,id in enumerate(NWM_data_gages) if id in NWM_cal_basin_gage]


gage_info =pd.read_csv( '/projects/mhpi/data/MERIT/gage_information/formatted_gage_csvs/all_gages_info_combined.csv')
gage_info = gage_info.sort_values(by='STAID')


all_time_range = pd.date_range(start=f'{1980}-01-01', end=f'{2019}-12-31', freq='D')
training_time_range = pd.date_range(start=f'{1980}-10-01', end=f'{1995}-09-30', freq='D')

train_idx_start = all_time_range.get_loc(training_time_range[0])
train_idx_end = all_time_range.get_loc(training_time_range[-1])+1

cal_idx_start = all_time_range.get_loc('2008-10-01')
cal_idx_end = all_time_range.get_loc('2013-09-30')+1


basin_area = np.expand_dims(obs_gage_root['attr']['area'][:], axis = -1 )

streamflow = obs_gage_root['forcing']['discharge'][:,:] 

streamflow = np.expand_dims(streamflow, axis = -1)


streamflow_trans = scale._basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


nan_threshold = 0.30

# Compute the number of NaNs per row
nan_counts = np.isnan(streamflow_trans[:,:,0]).sum(axis=1)

# Compute the percentage of NaNs per row
nan_percentage = nan_counts / streamflow_trans.shape[1]

# Find rows where NaNs are more than 30%
rows_with_high_nan_percentage = nan_percentage > nan_threshold

# Set these rows to all NaNs
streamflow_trans[rows_with_high_nan_percentage, :,0] = np.nan



small_area_indx = np.where(basin_area<200)[0]
small_untrained_idx = [x for x in small_area_indx if (x not in idx_train) and (x not in idx_cal) and not rows_with_high_nan_percentage[x]]
# small_untrained_idx=[x for x in range(len(streamflow_trans)) ]
streamflow_trans[np.array(idx_train),train_idx_start:train_idx_end,0] = np.nan
streamflow_trans[np.array(idx_cal),cal_idx_start:cal_idx_end,0] = np.nan
rootOut = "/projects/mhpi/yxs275/model/water_loss_model/"+'/dPL_local_daymet_new_attr_water_loss_v6v18_random_batch_filled_data_dynamic_remove_small_area/'
out = os.path.join(rootOut, "exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test/")  # output folder to save results


dHBV_simulation_root_1 = zarr.open_group(out+'/NWM_gage_simulation_1980_2000_warmuped', mode = 'r')
dHBV_simulation_1 = dHBV_simulation_root_1['Qs'][:]
dHBV_sim_gage = dHBV_simulation_root_1["COMID"][:]

dHBV_sim_gage_int = [int(x) for x in dHBV_sim_gage]
filtered_gages = gage_info[gage_info['STAID'].isin(dHBV_sim_gage_int)]

start_year = 1997
end_year = 2020
test_span = pd.date_range(f'{start_year}-01-01',f'{end_year}-12-31', freq='d')

dHBV_simulation_root_2 = zarr.open_group(out+'NWM_gage_simulation_1997_2020_warmuped', mode = 'r')
dHBV_simulation_2 = dHBV_simulation_root_2['Qs'][:,test_span.get_loc('2001-01-01'):test_span.get_loc(all_time_range[-1])+1]

dHBV_simulation = np.concatenate((dHBV_simulation_1,dHBV_simulation_2),axis = -1)


NWM_simulation = np.full((dHBV_simulation.shape),np.nan)


NWM_path = '/projects/mhpi/data/NWM/noaa-nwm-retrospective-3-0-pds/CONUS/daily_simulation_updated/'


NWM_timespan = pd.date_range('1979-02-01',f'2023-02-01', freq='d')

NWM_start = NWM_timespan.get_loc(all_time_range[0])
NWM_end = NWM_timespan.get_loc(all_time_range[-1])+1

valid_idx = []

lat = []
lon = []
area = []
for gageidx, gage in enumerate(dHBV_sim_gage):
    lat.append(gage_info[gage_info['STAID']==int(gage)]['LAT_GAGE'].values[0])
    lon.append(gage_info[gage_info['STAID']==int(gage)]['LNG_GAGE'].values[0])
    area.append(gage_info[gage_info['STAID']==int(gage)]['DRAIN_SQKM'].values[0])
    gage = gage.zfill(8)
    try:
        NWM_gage_root = zarr.open_group(NWM_path+gage, mode = 'r')
        NWM_simulation[gageidx,:] = NWM_gage_root['Qs_NWM'][0,NWM_start:NWM_end]/0.0283168 
        valid_idx.append(gageidx)
    except:
        print("gage ", gage, 'does not exist')

NWM_runoff = scale._basin_norm(
                        np.expand_dims(NWM_simulation,axis = -1) , basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day





evaDict = [stat.statError(NWM_runoff[small_untrained_idx,:,0], streamflow_trans[small_untrained_idx,:,0])]
evaDictLst = evaDict
keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','NNSE', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox_NWM = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox_NWM.append(temp)


print("NWM model'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','NNSE', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox_NWM[0][0]),
      np.nanmedian(dataBox_NWM[1][0]), np.nanmedian(dataBox_NWM[2][0]), np.nanmedian(dataBox_NWM[3][0]),
      np.nanmedian(dataBox_NWM[4][0]), np.nanmedian(dataBox_NWM[5][0]), np.nanmedian(dataBox_NWM[6][0]), 
      np.nanmedian(dataBox_NWM[7][0]), np.nanmedian(dataBox_NWM[8][0]),np.nanmedian(dataBox_NWM[9][0]),np.nanmedian(dataBox_NWM[10][0]))


evaDict = [stat.statError(dHBV_simulation[small_untrained_idx,:], streamflow_trans[small_untrained_idx,:,0])]
evaDictLst = evaDict
keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','NNSE', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox_dHBV = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox_dHBV.append(temp)


print("dHBV model'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','NNSE', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox_dHBV[0][0]),
      np.nanmedian(dataBox_dHBV[1][0]), np.nanmedian(dataBox_dHBV[2][0]), np.nanmedian(dataBox_dHBV[3][0]),
      np.nanmedian(dataBox_dHBV[4][0]), np.nanmedian(dataBox_dHBV[5][0]), np.nanmedian(dataBox_dHBV[6][0]),
      np.nanmedian(dataBox_dHBV[7][0]), np.nanmedian(dataBox_dHBV[8][0]),np.nanmedian(dataBox_dHBV[9][0]),np.nanmedian(dataBox_dHBV[10][0]))



rootOut = "/projects/mhpi/yxs275/model/"+'LSTM_local_daymet_filled_withNaN_NSE_with_same_forcing_HBV_remove_small_basins/'
out = os.path.join(rootOut, "exp_EPOCH300_BS100_RHO365_HS512_trainBuff365/")  # output folder to save results

LSTM_simulation_root = zarr.open_group(out+'/NWM_simulation', mode = 'r')

LSTM_simulation = LSTM_simulation_root['LSTM_Qs'][:]


evaDict = [stat.statError(LSTM_simulation[small_untrained_idx,:], streamflow_trans[small_untrained_idx,:,0])]
evaDictLst = evaDict
keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','NNSE', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox_LSTM = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox_LSTM.append(temp)


print("LSTM model'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','NNSE', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox_LSTM[0][0]),
      np.nanmedian(dataBox_LSTM[1][0]), np.nanmedian(dataBox_LSTM[2][0]), np.nanmedian(dataBox_LSTM[3][0]),
      np.nanmedian(dataBox_LSTM[4][0]), np.nanmedian(dataBox_LSTM[5][0]), np.nanmedian(dataBox_LSTM[6][0]),
      np.nanmedian(dataBox_LSTM[7][0]), np.nanmedian(dataBox_LSTM[8][0]),np.nanmedian(dataBox_LSTM[9][0]),np.nanmedian(dataBox_dHBV[10][0]))




Routed_simulation_root_zone = zarr.open_group('/projects/mhpi/yxs275/DM_output/NWM/'+'routed_simulation', mode = 'r')
Routed_simulation = Routed_simulation_root_zone['Qr'][:,:]





evaDict = [stat.statError(Routed_simulation[small_untrained_idx,:], streamflow_trans[small_untrained_idx,:,0])]
evaDictLst = evaDict
keyLst = ['NSE', 'KGE','CorrSp', 'Bias_rel','Corr','NNSE', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox_Routed = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox_Routed.append(temp)


print("Routed dHBV model'NSE', 'KGE','CorrSp', 'Bias_rel','Corr','NNSE', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox_Routed[0][0]),
      np.nanmedian(dataBox_Routed[1][0]), np.nanmedian(dataBox_Routed[2][0]), np.nanmedian(dataBox_Routed[3][0]),
      np.nanmedian(dataBox_Routed[4][0]), np.nanmedian(dataBox_Routed[5][0]), np.nanmedian(dataBox_Routed[6][0]),
      np.nanmedian(dataBox_Routed[7][0]), np.nanmedian(dataBox_Routed[8][0]),np.nanmedian(dataBox_Routed[9][0]),np.nanmedian(dataBox_Routed[10][0]))








model_name = 'NWM'
# title = 'LSTM performance'
#model_name = 'Routed'
title = 'National Water Model 3.0 Performance'
# title = r'$\delta$HBV2.0 + $\delta$MC Performance'
# title = r'$\delta$HBV2.0 Performance'
CorrSp = dataBox_NWM[2][0]
Bias_rel = dataBox_NWM[3][0]
lowgood = np.where((CorrSp>=0.5) & (Bias_rel<=1))[0]
lowbad = np.where((CorrSp<0.5) & (Bias_rel<=1))[0]
highgood = np.where((CorrSp>=0.5) & (Bias_rel>1))[0]
highbad = np.where((CorrSp<0.5) & (Bias_rel>1))[0]
fontsize = 16
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(8, 7))  # Example size: 10 inches by 6 inches


plt.scatter(Bias_rel[lowgood], CorrSp[lowgood] , color='green',s=10,label = "low bias, good shape")
plt.scatter(Bias_rel[highgood], CorrSp[highgood] , color='blue',s=10,label = "high bias, good shape")
plt.scatter(Bias_rel[lowbad], CorrSp[lowbad] , color='purple',s=10,label = "low bias, bad shape")
plt.scatter(Bias_rel[highbad], CorrSp[highbad] , color='red',s=10,label = "high bias, bad shape")
ax.axhline(0.5, color='black', linewidth=0.8)
ax.axvline(1, color='black', linewidth=0.8)

validstation_number = np.count_nonzero(~np.isnan(Bias_rel))
ax.text(0.96, 0.04, f'{round(len(highbad)/validstation_number*100,0)}%', verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, color='k', fontsize=15)
ax.text(0.04, 0.96, f'{round(len(lowgood)/validstation_number*100,0)}%', verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, color='k', fontsize=15)
ax.text(0.96, 0.96, f'{round(len(highgood)/validstation_number*100,0)}%', verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes, color='k', fontsize=15)
ax.text(0.04, 0.04, f'{round(len(lowbad)/validstation_number*100,0)}%', verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, color='k', fontsize=15)

lat_selected = np.array(lat)
lon_selected = np.array(lon)
# Adding labels and title
plt.ylim([-1.05,1.05])
plt.xlim([-0.2,10.2])

ax.set_title(title)
ax.set_xlabel('Absolute Relative Bias total flow')
ax.set_ylabel("Spearman's rho")
plt.legend(loc='upper center', frameon=True, ncol=1, bbox_to_anchor=(0.5,0.3),fontsize = fontsize)
plt.tight_layout()

plt.savefig("/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/" + f"spr_vs_bias_{model_name}.png", dpi=300)
# Show the plot
plt.show(block=True)


# model_name = 'LSTM'
# title = r'Distribution of NNSE (LSTM)'

model_name = 'NWM'
# title = r'Distribution of NNSE ($\delta$HBV2.0 + $\delta$MC)'
# title = r'Distribution of NNSE ($\delta$HBV2.0)'
title = r'Distribution of NNSE (NWM3.0)'

NNSE = dataBox_NWM[5][0]


fontsize = 18
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(11, 6))  # Example size: 10 inches by 6 inches


# Define the bins for categorization
# Define the bins for categorization
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Use numpy to categorize each correlation into bins and count them
counts, _ = np.histogram(NNSE, bins)

# Calculate the percentage of sites with correlation >= 0.8
percentage_above_08 = (NNSE >= 0.6).sum() / len(NNSE) * 100

# Bin labels for the plot
bin_labels = ['(0,0.2)', '(0.2,0.4)', '(0.4,0.6)', '(0.6,0.8)', '(0.8,1)']

# Create a bar chart
colors = ['aliceblue', 'lightblue', 'mediumpurple', 'blueviolet','purple']  # Define your colors here
plt.bar(bin_labels, counts, color=colors)

# Create a legend
colors = {'(0,0.2]':'aliceblue', '(0.2,0.4]':'lightblue', '(0.4,0.6]':'mediumpurple', '(0.6,0.8]':'blueviolet', '(0.8,1]':'purple'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, title="NNSE", loc='upper left', bbox_to_anchor=(1.05, 0.8))

# Add title and labels
plt.title(title)
plt.xlabel('NNSE')
plt.ylabel('Site Count')
plt.ylim([0,3000])

# Adding the annotation for the percentage of sites with cor >= 0.8

plt.text(-0.3, 2500, f'{percentage_above_08:.0f}% have NNSE >= 0.6',
          color='brown', fontweight='bold',clip_on=False,fontsize = 24)
plt.tight_layout()
plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/' + f"NNSE_{model_name}.png", dpi=300)

# Display the plot
plt.show(block=True)






model_name = 'LSTM'
label = r'NSE map of LSTM'
#label = r'NSE map of $\delta$HBV2.0'
# label = r'NSE map of $\delta$HBV2.0 + $\delta$MC'
# label = 'NSE map of National Water Model 3.0'
#title = r'Distribution of NNSE (NWM3.0)'
#title = r'$\delta$ model Performance'
NSE = dataBox_LSTM[0][0]

fontsize = 30
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(15, 10)) 
# Create a Basemap instance
m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50,
            llcrnrlon=-125, urcrnrlon=-65, resolution='i')

m.drawcoastlines()
m.drawcountries()
m.drawstates()


# Convert latitude and longitude to x and y coordinates
#x, y = m(lon[largeBasins], lat[largeBasins])
x, y = m(lon, lat)
# Plot each point, with size and color determined by mean flowrate and correlation

scatter = m.scatter(x, y, s=50, c=NSE, cmap=plt.cm.seismic,vmin=-0.2, vmax=1)
# Create an axes for the colorbar
#cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])

# Create a colorbar in the specified axes
plt.colorbar(scatter, pad=0.05,fraction = 0.11, location='bottom',label = label)

plt.tight_layout()

plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/' + f"NSE_map_{model_name}.png", dpi=300)


print("Done")




model_name = 'routeddHBV-NWM'
#label = r'NSE map of $\delta$HBV2.0'
label = r'$\delta$NSE map ($\delta$HBV2.0 + $\delta$MC - National Water Model 3.0)'
#title = r'Distribution of NNSE (NWM3.0)'
#title = r'$\delta$ model Performance'
NSE = dataBox_Routed[0][0] - dataBox_NWM[0][0]

fontsize = 30
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(15, 10)) 
# Create a Basemap instance
m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50,
            llcrnrlon=-125, urcrnrlon=-65, resolution='i')

m.drawcoastlines()
m.drawcountries()
m.drawstates()


# Convert latitude and longitude to x and y coordinates
#x, y = m(lon[largeBasins], lat[largeBasins])
x, y = m(lon, lat)
# Plot each point, with size and color determined by mean flowrate and correlation

scatter = m.scatter(x, y, s=50, c=NSE, cmap=plt.cm.seismic,vmin=-1, vmax=1)
# Create an axes for the colorbar
#cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])

# Create a colorbar in the specified axes
plt.colorbar(scatter, pad=0.05,fraction = 0.11, location='bottom',label = label)

plt.tight_layout()

plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/' + f"NSE_map_{model_name}.png", dpi=300)


print("Done")



legendLst = [ r'LSTM',r'National Water Model 3.0',r'$\delta$HBV2.0','$\delta$HBV2.0 + $\delta$MC']

fig,ax=plot.plotCDF([dataBox_LSTM[0][0],dataBox_NWM[0][0],dataBox_dHBV[0][0],dataBox_Routed[0][0]],
            ax=None,
            title=None,
            legendLst= legendLst,
            figsize=(8, 6.5),
            ref='121',
            #cLst= ['k','orange','y',"r","m","royalblue","b"] ,
            cLst=['k',"r",'orange','g',"b"],
            xlabel='Nash-Sutcliffe Model Efficiency (NSE)',
            ylabel='Cumulative Distribution Function (CDF)',
            showDiff='None',
            fontsize=16,
            xlim=[-1,1],
            linespec=["-", '--','--','--'])
            #linespec= ["-","-",'-','-','--','--','--','--'])


plt.annotate("", xy=(0.15, 0.4), xytext=(-0.05, 0.4), arrowprops=dict(headwidth=7, headlength=15, width=2,ec = "r",fc = "r"))
ax.text(-0.05, 0.35, "better")

plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/' + f"NSE_CDF.png", dpi=300)
plt.show(block=True)






nbin = 5
lower_bound = 0
upper_bound = 32000
#bins = np.linspace(lower_bound, upper_bound, nbin + 1)
bin_length = (upper_bound - lower_bound) / (nbin-1)
bins =np.array([0,8000,16000,24000,32000,100000])
bins_split =np.array([0,1000,3000,5000,10000,50000])
#area_bin_index_large_gage = np.digitize(gage_area_selected_large_gage[:,0], bins_split)
area_bin_index = np.digitize(basin_area[:,0], bins_split)
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
labels = []
for bin_i in range(len(bins)-1):
    labels.append(f'{bins_split[bin_i]}~{bins_split[bin_i+1]}')

plot1 = ax.boxplot( [ dataBox_LSTM[0][0][np.where(area_bin_index == i)][~np.isnan(dataBox_LSTM[0][0][np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+2*bin_length/7.0,patch_artist=True,boxprops=dict(facecolor="pink", color="k"),widths = bin_length/7 )
plot2 = ax.boxplot( [ dataBox_NWM[0][0][np.where(area_bin_index == i)][~np.isnan(dataBox_NWM[0][0][np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+3*bin_length/7.0,patch_artist=True,boxprops=dict(facecolor="red", color="k") ,widths = bin_length/7)
plot3 = ax.boxplot( [ dataBox_dHBV[0][0][np.where(area_bin_index == i)][~np.isnan(dataBox_dHBV[0][0][np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+4*bin_length/7.0,patch_artist=True,boxprops=dict(facecolor="mediumpurple", color="k"),widths = bin_length/7 )
plot4 = ax.boxplot( [ dataBox_Routed[0][0][np.where(area_bin_index == i)][~np.isnan(dataBox_Routed[0][0][np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+5*bin_length/7.0,patch_artist=True,boxprops=dict(facecolor="blue", color="k"),widths = bin_length/7 )

for whisker in plot1['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot1['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot1['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot1['medians']:
    median.set(ls='-', linewidth=2,color = "k")
for whisker in plot2['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot2['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot2['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot2['medians']:
    median.set(ls='-', linewidth=2,color = "k")
for whisker in plot3['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot3['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot3['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot3['medians']:
    median.set(ls='-', linewidth=2,color = "k")
for whisker in plot4['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot4['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot4['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot4['medians']:
    median.set(ls='-', linewidth=2,color = "k")




y_upper = 1.2
y_lower = -1
yrange = y_upper-y_lower
for i in range(1,nbin+1):

    num = len(basin_area[:,0][np.where(area_bin_index == i)])
    ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.1*(y_upper-y_lower), f'{num} sites')

# ax.add_patch( Rectangle(( 700, y_lower+0.5*yrange),500, yrange*0.05,  fc = "aliceblue",  ec ='k',ls = "-" , lw = 2) )
# ax.text(1300,y_lower+0.5*yrange, r"direct forward v3 (all model at gage_level)")

ax.add_patch( Rectangle(( 700, y_lower+0.4*yrange), 500, yrange*0.05,  fc = "pink",  ec ='k',ls = "-" , lw = 2) )
ax.text(1300, y_lower+0.4*yrange, r"LSTM")

ax.add_patch( Rectangle(( 700, y_lower+0.3*yrange), 500, yrange*0.05,  fc = "red",  ec ='k',ls = "-" , lw = 2) )
ax.text(1300, y_lower+0.3*yrange, r'National Water Model 3.0')

ax.add_patch( Rectangle(( 700, y_lower+0.2*yrange), 500, yrange*0.05,  fc = "mediumpurple",  ec ='k',ls = "-" , lw = 2) )
ax.text(1300, y_lower+0.2*yrange, r'$\delta$HBV2.0')


ax.add_patch( Rectangle(( 700, y_lower+0.1*yrange), 500, yrange*0.05,  fc = "blue",  ec ='k',ls = "-" , lw = 2) )
ax.text(1300, y_lower+0.1*yrange, r'$\delta$HBV2.0 + $\delta$MC')



ax.set_ylabel("Daily NSE")
ax.set_xlabel(r"Drainage area (km$^2$)")

ax.set_yticks(np.arange(y_lower,y_upper,0.2))
ax.set_ylim([y_lower,y_upper])
ax.set_xlim([lower_bound,upper_bound+bin_length])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.vlines([-0.5,0,0.5], -2, 4,color ="k",linestyles='--',lw = 1.5)
ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), y_lower,y_upper,color ="k",linestyles='--',lw = 2.5)
tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
ax.set_xticks(tick_positions)
#ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
ax.set_xticklabels(labels)

plt.savefig('/projects/mhpi/yxs275/model/dPLHBVrelease-master/hydroDL-dev/example/NWM/'+"boxplot_NSE_area.png", dpi=300)
plt.show(block=True)

print("Done")