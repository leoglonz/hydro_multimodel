from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats
from hydroDL.post import  stat
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle

def bias_meanflowratio_calc(pred,target):
    ngrid,nt = pred.shape    
    Bias = np.full(ngrid, np.nan)
    meanflowratio = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            Bias[k] = (np.sum(xx)-np.sum(yy))/(np.sum(yy)+0.00001)
            meanflowratio[k]  = np.sum(xx)/(np.sum(yy)+0.00001)

    return Bias, meanflowratio

def annual_bias_meanflowratio_calc(pred,target, yearstart, yearsend, time_allyear):
    Bias_ = 0

    mean_ = 0

    for year in range(yearstart,yearsend):
        time_year = pd.date_range(f'{year}-10-01', f'{year+1}-09-30', freq='d')
        idx_start = time_allyear.get_loc(time_year[0])
        idx_end = time_allyear.get_loc(time_year[-1])

        year_Bias_,year_mean_ = bias_meanflowratio_calc(pred[:,idx_start:idx_end+1],target[:,idx_start:idx_end+1])

        Bias_ = Bias_ + year_Bias_

        mean_ = mean_+year_mean_

    nyear = yearsend-yearstart
    Bias_ = Bias_/nyear


    mean_ = mean_/nyear




##Read the area
attribute_file = "/projects/mhpi/yxs275/Data/attributes_haoyu/attributes_haoyu.csv"
attributes = pd.read_csv(attribute_file)
area_all = attributes['area'].values
gageid_all = attributes['id'].values

##Read Routing simulations
#path_to_zarr_routed = Path("/projects/mhpi/tbindas/dMC-dev/runs/trained_models/2024-03-14_05-31-55/zarr_data/1981-10-01_1995-09-30_validation_data")
#path_to_zarr_routed = Path("/projects/mhpi/tbindas/dMC-dev/runs/trained_models/dMC_v0.4_2024-03-20_19-29-51_merit_conus_v1.1/zarr_data/1981-10-01_1995-09-30_validation")
path_to_zarr_routed = Path("/projects/mhpi/hjj5218/project/01.dMC-dev/runs/dMCv1.0.1-train_test_gages_3000-merit_conus_v3.0/2024-04-18_14-21-10/zarr_data/zones_74_73_77_78_72_75_71_1981-10-01_1995-09-30_validation")
ds_routed = xr.open_zarr(path_to_zarr_routed)
gage_ids = ds_routed.gage_ids.data

gage_ids_filled = [id.zfill(8) for id in gage_ids]


path_to_zarr_routed_global = Path("/projects/mhpi/tbindas/dMC-dev/runs/trained_models/dMC_v0.4_2024-03-20_19-29-16_merit_global_v1.1/zarr_data/1981-10-01_1995-09-30_validation")
ds_routed_global = xr.open_zarr(path_to_zarr_routed_global)

##Read forward simulations

data_folder = "/projects/mhpi/yxs275/DM_output/plot/"
time_routed = pd.date_range('1981-10-01',f'1995-09-29', freq='d')

path_to_zarr_forward = Path("/projects/mhpi/yxs275/Data/zarr/CONUS3200")

ds_forward = xr.open_zarr(path_to_zarr_forward)
gageIDs = ds_forward.gageID.data


[C, Ind, SubInd] = np.intersect1d(gage_ids_filled, gageIDs, return_indices=True)


selected_gage_in_routing_simulations = gage_ids[Ind]
selected_gage_in_forward_simulations = gageIDs[SubInd]


Routed_simulation_ds_global = ds_routed_global.predictions.sel(gage_ids = selected_gage_in_routing_simulations).groupby(ds_routed_global.predictions.time.dt.floor('D'))
Routed_simulation_global = Routed_simulation_ds_global.mean().values
Routed_simulation_global = Routed_simulation_global/0.0283168 
Routed_observations_ds_global = ds_routed_global.observations.sel(gage_ids = selected_gage_in_routing_simulations).groupby(ds_routed_global.observations.time.dt.floor('D'))
Routed_observations_global = Routed_observations_ds_global.mean().values 
Routed_observations_global = Routed_observations_global/0.0283168 


Routed_simulation_ds = ds_routed.predictions.sel(gage_ids = selected_gage_in_routing_simulations).groupby(ds_routed.predictions.time.dt.floor('D'))
Routed_simulation = Routed_simulation_ds.mean().values
Routed_simulation = Routed_simulation/0.0283168 
Routed_observations_ds = ds_routed.observations.sel(gage_ids = selected_gage_in_routing_simulations).groupby(ds_routed.observations.time.dt.floor('D'))
Routed_observations = Routed_observations_ds.mean().values 
Routed_observations = Routed_observations/0.0283168 


forward_simulation = ds_forward['data'].sel(Variables="forward",gageID=selected_gage_in_forward_simulations ,time =time_routed ).values
observation = ds_forward['data'].sel(Variables="observations",gageID=selected_gage_in_forward_simulations ,time =time_routed).values


index_in_all = [np.where(gageid_all==int(x))[0][0] for x in selected_gage_in_routing_simulations]
area_selected = area_all[index_in_all]
gageid_all[index_in_all]

aridity = attributes['aridity']
aridity_selected = aridity[index_in_all]

## Evaluate metrics
evaDict = [stat.statError(forward_simulation[:,:], observation[:,:])]
evaDictLst = evaDict
keyLst = ['NSE', 'Bias','Corr','Bias_rel','NNSE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)

NNSE_forward_simulation = dataBox[4][0]
Bias_forward_simulation = dataBox[1][0]

print("Median NNSE of forward simulation is ", np.nanmedian(NNSE_forward_simulation))

evaDict = [stat.statError(Routed_simulation[:,:], Routed_observations[:,:])]
evaDictLst = evaDict
keyLst = ['NSE', 'Bias','Corr','Bias_rel','NNSE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)

NNSE_routed = dataBox[4][0]
Bias_routed = dataBox[1][0]
print("Median NNSE of routed simulation/continental is ", np.nanmedian(NNSE_routed))

evaDict = [stat.statError(Routed_simulation_global[:,:], Routed_observations_global[:,:])]
evaDictLst = evaDict
keyLst = ['NSE', 'Bias','Corr','Bias_rel','NNSE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
NNSE_routed_global = dataBox[4][0]
Bias_routed_global = dataBox[1][0]
print("Median NNSE of routed simulation/global is ", np.nanmedian(NNSE_routed_global))



evaDict = [stat.statError(Routed_simulation[:,:], forward_simulation[:,:])]
evaDictLst = evaDict
keyLst = ['NSE', 'Bias','Corr','Bias_rel','NNSE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)

NNSE_routed_forward = dataBox[4][0]
Bias_routed_forward = dataBox[1][0]
print("Median NNSE of routed/forward is ", np.nanmedian(NNSE_routed_forward))


##Plotting

nbin = 5
lower_bound = 0
upper_bound = 32000
#bins = np.linspace(lower_bound, upper_bound, nbin + 1)
bin_length = (upper_bound - lower_bound) / (nbin-1)
bins =np.array([0,8000,16000,24000,32000,100000])
bins_split =np.array([0,500,1000,2000,5000,1000000])


area_selected =np.array(area_selected)
area_bin_index = np.digitize(area_selected, bins_split)
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
labels = []
for bin_i in range(len(bins)-1):
    labels.append(f'{bins_split[bin_i]}~{bins_split[bin_i+1]}')

plot1 = ax.boxplot( [ NNSE_forward_simulation[np.where(area_bin_index == i)][~np.isnan(NNSE_forward_simulation[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+1*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="aliceblue", color="k") ,widths = bin_length/6)
plot2 = ax.boxplot( [ NNSE_routed[np.where(area_bin_index == i)][~np.isnan(NNSE_routed[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+2*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="mediumpurple", color="k"),widths = bin_length/6 )
plot3 = ax.boxplot( [ NNSE_routed_global[np.where(area_bin_index == i)][~np.isnan(NNSE_routed_global[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+3*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="pink", color="k"),widths = bin_length/6 )

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




y_upper = 1.2
y_lower = 0
yrange = y_upper-y_lower
for i in range(1,nbin+1):

    num = len(area_selected[np.where(area_bin_index == i)])
    ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.1*(y_upper-y_lower), f'{num} sites')

ax.add_patch( Rectangle(( 700, y_lower+0.3*yrange),200, yrange*0.05,  fc = "aliceblue",  ec ='k',ls = "-" , lw = 2) )
ax.text(1000,y_lower+0.3*yrange, r"Forward $ft^3/s$")
ax.add_patch( Rectangle(( 700, y_lower+0.2*yrange), 200, yrange*0.05,  fc = "mediumpurple",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.2*yrange, r"Routed flow/continental")
ax.add_patch( Rectangle(( 700, y_lower+0.1*yrange), 200, yrange*0.05,  fc = "pink",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.1*yrange, r"Routed flow/global")

ax.set_ylabel("$NNSE$")
ax.set_xlabel(r"Drainage area (km$^2$)")

ax.set_yticks(np.arange(y_lower,y_upper,yrange/6))
ax.set_ylim([y_lower,y_upper])
ax.set_xlim([lower_bound,upper_bound+bin_length])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.vlines([-0.5,0,0.5], -2, 4,color ="k",linestyles='--',lw = 1.5)
ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), y_lower,y_upper,color ="k",linestyles='--',lw = 2.5)
tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
ax.set_xticks(tick_positions)
#ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
ax.set_xticklabels(labels)

plt.savefig("boxplot_NSE_area.png", dpi=300)
plt.show(block=True)

print("Done")






nbin = 5
lower_bound = 0
upper_bound = 32000
#bins = np.linspace(lower_bound, upper_bound, nbin + 1)
bin_length = (upper_bound - lower_bound) / (nbin-1)
bins =np.array([0,8000,16000,24000,32000,100000])
bins_split =np.array([0,500,1000,2000,5000,1000000])


area_selected =np.array(area_selected)
area_bin_index = np.digitize(area_selected, bins_split)
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
labels = []
for bin_i in range(len(bins)-1):
    labels.append(f'{bins_split[bin_i]}~{bins_split[bin_i+1]}')

plot1 = ax.boxplot( [ Bias_forward_simulation[np.where(area_bin_index == i)][~np.isnan(Bias_forward_simulation[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ],whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+1*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="aliceblue", color="k") ,widths = bin_length/6)
plot2 = ax.boxplot( [ Bias_routed[np.where(area_bin_index == i)][~np.isnan(Bias_routed[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ],whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+2*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="mediumpurple", color="k"),widths = bin_length/6 )
plot3 = ax.boxplot( [ Bias_routed_global[np.where(area_bin_index == i)][~np.isnan(Bias_routed_global[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ],whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+3*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="pink", color="k"),widths = bin_length/6 )

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




y_upper = 5
y_lower = -5
yrange = y_upper-y_lower
for i in range(1,nbin+1):

    num = len(area_selected[np.where(area_bin_index == i)])
    ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.1*(y_upper-y_lower), f'{num} sites')

ax.add_patch( Rectangle(( 700, y_lower+0.3*yrange),200, yrange*0.05,  fc = "aliceblue",  ec ='k',ls = "-" , lw = 2) )
ax.text(1000,y_lower+0.3*yrange, r"Forward $ft^3/s$")
ax.add_patch( Rectangle(( 700, y_lower+0.2*yrange), 200, yrange*0.05,  fc = "mediumpurple",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.2*yrange, r"Routed flow/continental")
ax.add_patch( Rectangle(( 700, y_lower+0.1*yrange), 200, yrange*0.05,  fc = "pink",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.1*yrange, r"Routed flow/global")

ax.set_ylabel("Total bias percentage")
ax.set_xlabel(r"Drainage area (km$^2$)")

ax.set_yticks(np.arange(y_lower,y_upper,yrange/10))
ax.set_ylim([y_lower,y_upper])
ax.set_xlim([lower_bound,upper_bound+bin_length])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.vlines([-0.5,0,0.5], -2, 4,color ="k",linestyles='--',lw = 1.5)
ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), y_lower,y_upper,color ="k",linestyles='--',lw = 2.5)
tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
ax.set_xticks(tick_positions)
#ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
ax.set_xticklabels(labels)

plt.savefig("boxplot_Bias_area.png", dpi=300)
plt.show(block=True)

print("Done")

















nbin = 4
lower_bound = 0
upper_bound = 24000
#bins = np.linspace(lower_bound, upper_bound, nbin + 1)
bin_length = (upper_bound - lower_bound) / (nbin-1)
bins =np.array([0,8000,16000,24000,32000])

bins_split =np.array([0,0.8,1.2,2.5,5])


aridity_selected =np.array(aridity_selected)
area_bin_index = np.digitize(aridity_selected, bins_split)
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
labels = []
for bin_i in range(len(bins_split)-1):
    labels.append(f'{bins_split[bin_i]}~{bins_split[bin_i+1]}')

plot1 = ax.boxplot( [ Bias_forward_simulation[np.where(area_bin_index == i)][~np.isnan(Bias_forward_simulation[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+1*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="aliceblue", color="k") ,widths = bin_length/6)
plot2 = ax.boxplot( [ Bias_routed[np.where(area_bin_index == i)][~np.isnan(Bias_routed[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+2*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="mediumpurple", color="k"),widths = bin_length/6 )
plot3 = ax.boxplot( [ Bias_routed_global[np.where(area_bin_index == i)][~np.isnan(Bias_routed_global[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+3*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="pink", color="k"),widths = bin_length/6 )

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


y_upper = 5
y_lower = -5
yrange = y_upper-y_lower
for i in range(1,nbin+1):

    num = len(area_selected[np.where(area_bin_index == i)])
    ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.1*(y_upper-y_lower), f'{num} sites')

ax.add_patch( Rectangle(( 700, y_lower+0.3*yrange),200, yrange*0.05,  fc = "aliceblue",  ec ='k',ls = "-" , lw = 2) )
ax.text(1000,y_lower+0.3*yrange, r"Forward $ft^3/s$")
ax.add_patch( Rectangle(( 700, y_lower+0.2*yrange), 200, yrange*0.05,  fc = "mediumpurple",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.2*yrange, r"Routed flow/continental")
ax.add_patch( Rectangle(( 700, y_lower+0.1*yrange), 200, yrange*0.05,  fc = "pink",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.1*yrange, r"Routed flow/global")

ax.set_ylabel("Total bias percentage")
ax.set_xlabel(r"Aridity")

ax.set_yticks(np.arange(y_lower,y_upper,yrange/10))
ax.set_ylim([y_lower,y_upper])
ax.set_xlim([lower_bound,upper_bound+bin_length])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.vlines([-0.5,0,0.5], -2, 4,color ="k",linestyles='--',lw = 1.5)
ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), y_lower,y_upper,color ="k",linestyles='--',lw = 2.5)
tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
ax.set_xticks(tick_positions)
#ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
ax.set_xticklabels(labels)

plt.savefig("boxplot_Bias_aridity.png", dpi=300)
plt.show(block=True)

print("Done")

lat = attributes["lat"].values
lon = attributes["lon"].values
lat = lat[index_in_all]
lon = lon[index_in_all]

#largeBasins=np.where(area_selected>24000)[0]
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# Load your dataset here, which must include latitude, longitude, correlation, and mean flowrate.
fontsize = 18
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
delta_Bias = (np.sum(Routed_simulation,axis = 1)-np.sum(Routed_observations,axis = 1))/np.sum(Routed_observations,axis = 1)
#delta_Bias=delta_Bias[largeBasins]
scatter = m.scatter(x, y, s=area_selected/100, c=delta_Bias, cmap='jet',vmin=-0.2, vmax=0.2)

# Create an axes for the colorbar
#cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])

# Create a colorbar in the specified axes
plt.colorbar(scatter, pad=0.05,fraction = 0.11, location='bottom', label=r'(total routed - total forward )/ total forward flow (area>24000km$^2$)')

plt.tight_layout()
plt.savefig("Map_area_bias_observation.png", dpi=300)


print("Done")


def bias_meanflowratio_calc(pred,target):
    ngrid,nt = pred.shape    
    Bias = np.full(ngrid, np.nan)
    meanflowratio = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            Bias[k] = (np.sum(xx)-np.sum(yy))/(np.sum(yy)+0.00001)
            meanflowratio[k]  = np.sum(xx)/np.sum(yy)


lat = attributes["lat"].values
lon = attributes["lon"].values
lat = lat[index_in_all]
lon = lon[index_in_all]

#largeBasins=np.where(area_selected>24000)[0]
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# Load your dataset here, which must include latitude, longitude, correlation, and mean flowrate.
fontsize = 18
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
delta_Bias,_ = bias_meanflowratio_calc(Routed_simulation,Routed_observations)
#delta_Bias=delta_Bias[largeBasins]
#scatter = m.scatter(x, y, s=area_selected/100, c=delta_Bias, cmap='jet',vmin=-0.2, vmax=0.2)
scatter = m.scatter(x, y, s=aridity_selected**2*100, c=delta_Bias, cmap='jet',vmin=-0.2, vmax=0.2)
# Create an axes for the colorbar
#cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])

# Create a colorbar in the specified axes
plt.colorbar(scatter, pad=0.05,fraction = 0.11, location='bottom', label=r'(total forward - total obs )/ total obs flow')

plt.tight_layout()
plt.savefig("Map_arid_bias_observation.png", dpi=300)


print("Done")