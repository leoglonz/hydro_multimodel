import numpy as np
import pandas as pd

import json
import glob
import sys
sys.path.append('../../')
from hydroDL.post import plot, stat

from mpl_toolkits import basemap


def _basin_norm(
    x: np.array, basin_area: np.array,  to_norm: bool
) -> np.array:
    """
    Normalize or denormalize streamflow data with basin area and mean precipitation.

    The formula is as follows when normalizing (denormalize equation is its inversion):

    .. math:: normalized_x = \frac{x}{area * precipitation}

    Because units of streamflow, area, and precipitation are ft^3/s, km^2 and mm/day, respectively,
    and we need (m^3/day)/(m^3/day), we transform the equation as the code shows.

    Parameters
    ----------
    x
        data to be normalized or denormalized
    basin_area
        basins' area
    mean_prep
        basins' mean precipitation
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    nd = len(x.shape)
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basin_area, (1, x.shape[1]))

    if to_norm is True:
        # flow = (x * 0.0283168 * 3600 * 24) / (
        #     (temparea * (10**6)) * (tempprep * 10 ** (-3))
        # )  # (m^3/day)/(m^3/day)

        flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

    else:
        # flow = (
        #     x
        #     * ((temparea * (10**6)) * (tempprep * 10 ** (-3)))
        #     / (0.0283168 * 3600 * 24)
        # )
        flow = (
            x
            * ((temparea * (10**6)) * (10 ** (-3)))
            / (0.0283168 * 3600 * 24)
        )

    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow




data_folder_3200 = "/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/gages/dataCONUS3200/"


with open(data_folder_3200+'train_data_dict.json') as f:
    train_data_dict = json.load(f)
smallBasinID = train_data_dict['sites_id']



data_folder = "/data/yxs275/CONUS_data/HUC10/version_1_11_2014_continental_routing/"

data_split_folder = "/data/yxs275/CONUS_data/HUC10/dPL_1_11_2024/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/basin_split/"
attributeALL_df = pd.read_csv(data_folder + "attributes.csv")
basinID_all = attributeALL_df.gage_ID.values
HUC10_area = attributeALL_df.area.values


GAGEII_folder = "/data/yxs275/CONUS_data/all_GAGEII/gages/dataGAGEall/"
GAGEII_flow =np.load(GAGEII_folder+"train_flow.npy")
GAGEII_attr = np.load(GAGEII_folder+"train_attr.npy")
GAGEII_forcing = np.load(GAGEII_folder+"train_forcing.npy")
with open(GAGEII_folder+'train_data_dict.json') as f:
    train_data_dict = json.load(f)

GAGEII_ID = train_data_dict['sites_id']
AllGageTime = pd.date_range(train_data_dict["t_final_range"][0], train_data_dict["t_final_range"][1], freq='d')
AllHUC10Time = pd.date_range('1980-01-01', f'2020-12-31', freq='d')
attributeAllLst  = train_data_dict['constant_cols']
GAGEIIAreaName = "DRAIN_SQKM"
GAGEII_area = GAGEII_attr[:,np.where(np.array(attributeAllLst)=="DRAIN_SQKM")[0]]
GAGEII_lat = GAGEII_attr[:,np.where(np.array(attributeAllLst)=="LAT_GAGE")[0]]
GAGEII_lon = GAGEII_attr[:,np.where(np.array(attributeAllLst)=="LNG_GAGE")[0]]
streamflow_trans = _basin_norm(
                        GAGEII_flow[:, :, 0 :  1].copy(), GAGEII_area, to_norm=True
                    )

selected_Basin = np.load(data_folder+"selected_Basin_matched_v3.npy")
selected_GAGE = np.load(data_folder+"selected_GAGE_matched_v3.npy")
# HUC10_forcing_selected = np.full((len(selected_Basin),GAGEII_flow.shape[1],GAGEII_forcing.shape[-1]),np.nan)
# GAGEII_forcing_selected = np.full((len(selected_Basin),GAGEII_flow.shape[1],GAGEII_forcing.shape[-1]),np.nan)


# forcing_split_folder = "/data/yxs275/CONUS_data/HUC10/version_2_11_25/forcing_split/"
Insmall_list = []
lat_selected = []
lon_selected = []
for idx, basinID in enumerate(selected_Basin):
    gageID = selected_GAGE[idx]
    gageIdx = np.where(np.array(GAGEII_ID) == gageID)[0][0]
    if gageID in smallBasinID:
        Insmall_list.append(gageID)


    try:
        basinIdx = np.where(np.array(basinID_all) == int(basinID))[0][0]
        print("Gage area ", GAGEII_area[gageIdx], "Basin area ", HUC10_area[basinIdx], )
        pred = np.load(data_split_folder + f"{basinID}.npy")[:,
               AllHUC10Time.get_loc(AllGageTime[0]):AllHUC10Time.get_loc(AllGageTime[-1])]
        obs = streamflow_trans[gageIdx: gageIdx + 1, :, 0]
        if idx == 0:

            predAll = pred
            obsAll = obs
        else:
            predAll = np.concatenate((predAll,pred),axis = 0)
            obsAll = np.concatenate((obsAll,obs),axis = 0)
        lat_selected.append(GAGEII_lat[gageIdx])
        lon_selected.append(GAGEII_lon[gageIdx])
    except:
        print(basinID, "is not selected now")
    # HUC10_forcing_selected[idx,:,:] = np.load(forcing_split_folder+f"{basinID}.npy")
    # GAGEII_forcing_selected[idx,:,:] = GAGEII_forcing[gageIdx,:,:]



# date_all = pd.date_range('1980-01-01',f'2000-01-01', freq='d', closed='left')
# for year in range(1980,2020):
#     date_year = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='d', closed='left')
#     startidx = date_all.get_loc(date_year[0])
#     endidx = date_all.get_loc(date_year[-1])
#     HUC10_forcing_selected_sliced = HUC10_forcing_selected[:,startidx:endidx+1,1]
#     GAGEII_forcing_selected_sliced = GAGEII_forcing_selected[:, startidx:endidx + 1, 1]
#     print("Year", year, "max off", np.nanmax(abs(GAGEII_forcing_selected_sliced-HUC10_forcing_selected_sliced)))

# new_HUC10_forcing_selected_1980 = np.load("/data/yxs275/CONUS_data/HUC10/version_2_11_25/forcings_slected_1980.npy")
# new_HUC10_forcing_selected_2000 = np.load("/data/yxs275/CONUS_data/HUC10/version_2_11_25/forcings_slected_2020.npy")
# new_HUC10_forcing_selected = np.concatenate((new_HUC10_forcing_selected_1980,new_HUC10_forcing_selected_2000),axis = 1)
# np.save( "/data/yxs275/CONUS_data/HUC10/version_2_11_25/HUC10_forcing_selected.npy", HUC10_forcing_selected)
# np.save( "/data/yxs275/CONUS_data/HUC10/version_2_11_25/GAGEII_forcing_selected.npy", GAGEII_forcing_selected)
#
evaDict = [stat.statError(predAll[:,365:], obsAll[:,365:])]
evaDictLst = evaDict
keyLst = ['NSE', 'Bias','Corr','Bias_rel']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)


print("LSTM model'NSE', 'Bias','CorrSp','Bias_rel",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]))

import matplotlib.pyplot as plt
import numpy as np
CorrSp = dataBox[2][0]
Bias_rel = dataBox[3][0]
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

lat_selected = np.array(lat_selected)
lon_selected = np.array(lon_selected)
# Adding labels and title
plt.ylim([-1.05,1.05])
plt.xlim([-0.2,10.2])

ax.set_title(r'$\delta$ model Performance')
ax.set_xlabel('Absolute Relative Bias total flow')
ax.set_ylabel("Spearman's rho")
plt.legend(loc='upper center', frameon=True, ncol=1, bbox_to_anchor=(0.5,0.3),fontsize = fontsize)
plt.tight_layout()

plt.savefig("/data/yxs275/CONUS_data/HUC10/dPL_1_11_2024/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/" + "spr_vs_bias.png", dpi=300)
# Show the plot
plt.show(block=True)

fontsize = 12
plt.rcParams.update({'font.size': fontsize})
cRange = [1,4]
vmin = cRange[0]
vmax = cRange[1]
fig = plt.figure(figsize=(8, 6))
ax = fig.subplots()
bounding = [np.min(lat_selected)-0.5, np.max(lat_selected)+0.5,
                    np.min(lon_selected)-0.5,np.max(lon_selected)+0.5]
prj='cyl'
mm = basemap.Basemap(
    llcrnrlat=bounding[0],
    urcrnrlat=bounding[1],
    llcrnrlon=bounding[2],
    urcrnrlon=bounding[3],
    projection=prj,
    resolution='c',
    ax=ax)
mm.drawcoastlines()
mm.drawstates(linestyle='dashed')
mm.drawcountries(linewidth=1.0, linestyle='-.')
x, y = mm(lon_selected[lowgood], lat_selected[lowgood])
mm.scatter(x, y ,marker='o', color='green',s = 10,label = "low bias, good shape")
x, y = mm(lon_selected[highgood], lat_selected[highgood])
mm.scatter(x, y ,marker='o', color='blue',s = 10,label = "high bias, good shape")
x, y = mm(lon_selected[lowbad], lat_selected[lowbad])
mm.scatter(x, y ,marker='o', color='purple',s = 10,label = "low bias, bad shape")
x, y = mm(lon_selected[highbad], lat_selected[highbad])
mm.scatter(x, y ,marker='o', color='red',s = 10,label = "high bias, bad shape")
ax.set_title(r'$\delta$ model Performance')
plt.legend(loc='upper center', frameon=True, ncol=2, bbox_to_anchor=(0.47,-0.05),fontsize = fontsize-2)
# mm.scatter(Bias_rel[highgood], CorrSp[highgood] , color='blue',s=10,label = "high bias, good shape")
# mm.scatter(Bias_rel[lowbad], CorrSp[lowbad] , color='purple',s=10,label = "low bias, bad shape")
# mm.scatter(Bias_rel[highbad], CorrSp[highbad] , color='red',s=10,label = "high bias, bad shape")
plt.savefig("/data/yxs275/CONUS_data/HUC10/dPL_1_11_2024/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/" + "spr_bias_map.png", dpi=300)
plt.show(block=True)
# Qr_folder = "/data/yxs275/CONUS_data/HUC10/dPL_version2/exp_EPOCH50_BS100_RHO365_HS256_trainBuff365/"
# attributeALL_df = pd.read_csv(data_folder + "attributes.csv")
# basinID = attributeALL_df.gage_ID.values
# batchSize = 1000
# iS = np.arange(0, len(basinID), batchSize)
# iE = np.append(iS[1:], len(basinID))
# for item in range(len(iS)):
#
#     Qr_Batch = pd.read_csv(Qr_folder + f"Qr_{iS[item]}_{iE[item]}", dtype=np.float32, header=None).values
#     attributeBatch_file = data_folder+f"attributes_{iS[item]}_{iE[item]}.csv"
#     attributeBatch_df = pd.read_csv(attributeBatch_file)
#     attributeBatch_ID = attributeBatch_df.gage_ID.values
#     for idx, ID in enumerate(attributeBatch_ID):
#         ID_str = str(ID).zfill(10)
#         Qr = Qr_Batch[idx:idx+1,:]
#         np.save(data_split_folder+f"{ID_str}.npy",Qr)


