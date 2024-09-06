import matplotlib.pyplot as plt
import numpy as np
import zarr
from datetime import datetime
import xarray as xr

import numpy as np
import pandas as pd

import json
import glob
import sys
sys.path.append('../../')
# from hydroDL.post import plot, stat

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

#data_split_folder = "/data/yxs275/CONUS_data/HUC10/dPL_02_01_2024_hs512_with_area/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/basin_split/"
#data_split_folder = "/data/yxs275/CONUS_data/HUC10/dPL_version3_12_5/exp_EPOCH50_BS100_RHO365_HS256_trainBuff365/basin_split/"
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


data_folder_new = "/data/yxs275/CONUS_data/HUC10/version_1_11_2014_continental_routing/"
selected_Basin = np.load(data_folder_new+"selected_Basin.npy")
selected_GAGE = np.load(data_folder_new+"selected_GAGE.npy")

Insmall_list = []
lat_selected = []
lon_selected = []
meanflowrate_selected = []
evaluateTime = pd.date_range('1981-01-01', f'2019-12-31', freq='d')
count = 0
Valid_data_percentage = []
HUC10_selected = []
GageII_selected = []

area_selected = []
HUC10area_selected = []





path_to_zarr = "/data/tkb5476/projects/dMC-dev/runs/dMCv0.1-srb_validation-dpl_v3/2024-02-08_14-09-48/1985-10-04_1995-09-30"
ds = xr.open_zarr(path_to_zarr)
gages_routed = ds.gage_ids.data
pred_routed = ds.predictions.resample(time='1D').mean().compute().data
obs_routed = ds.observations.resample(time='1D').mean().compute().data


for idx, basinID in enumerate(selected_Basin):
    gageID = selected_GAGE[idx]
    gageIdx = np.where(np.array(GAGEII_ID) == gageID)[0][0]
    # if gageID  in smallBasinID:
    #     Insmall_list.append(gageID)
    #     # continue
    #
    #
    # else:
    try:


        basinIdx = np.where(np.array(basinID_all) == int(basinID))[0][0]
        print("Gage area ", GAGEII_area[gageIdx],"Gage ID",gageID, "Basin area ", HUC10_area[basinIdx], "Basin ID", basinID )
        # pred = np.load(data_split_folder + f"{basinID}.npy")[:,
        #        AllHUC10Time.get_loc(evaluateTime[0]):AllHUC10Time.get_loc(evaluateTime[-1])]
        obs = streamflow_trans[gageIdx: gageIdx + 1, AllGageTime.get_loc(evaluateTime[0]):AllGageTime.get_loc(evaluateTime[-1]), 0]
        attribute = GAGEII_attr[gageIdx: gageIdx + 1,:] #attributeALL_df.values[basinIdx:basinIdx + 1, :]
        attribute_extracted =  attributeALL_df[attributeALL_df['gage_ID'] ==  int(basinID)].values[:,1:]
       # attribute = np.concatenate((np.expand_dims(gageID,axis = [0,1]), attribute),axis = 1)
        nan_count = np.isnan(obs).sum()
        print("Valid data percentage: ", nan_count/len(obs[0,:]))
        Valid_data_percentage.append(nan_count/len(obs[0,:]))
        if nan_count/len(obs[0,:]) <0.3:

            if str(int(gageID)) in list(gages_routed):
                routedTime = pd.date_range('1985-10-04', f'1995-10-01', freq='d')
                routedIdx = np.where(gages_routed == str(int(gageID)))[0][0]
                predar = pred_routed[routedIdx:routedIdx+1,:]#*(24*60*60)/(GAGEII_area[gageIdx,0] *1000)
                predrf = np.load(data_split_folder + f"{basinID}.npy")[:,
                       AllHUC10Time.get_loc(routedTime[0]):AllHUC10Time.get_loc(routedTime[-1])]* (GAGEII_area[gageIdx,0] *1000)/(24*60*60)         #* (HUC10_area[basinIdx] *1000)/(24*60*60)
                HUC10_selected.append(basinID)
                obs = streamflow_trans[gageIdx: gageIdx + 1, AllGageTime.get_loc(routedTime[0]):AllGageTime.get_loc(routedTime[-1]), 0] * (GAGEII_area[gageIdx,0] *1000)/(24*60*60)
                area_selected.append(GAGEII_area[gageIdx,0])
                HUC10area_selected.append(HUC10_area[basinIdx])
            #    obs1 =  obs_routed[routedIdx:routedIdx+1,:]* (24*60*60) /(GAGEII_area[gageIdx,0] *1000)
                if count == 0:

                    predAllar = predar
                    predAllrf = predrf
                    obsAll = obs
                    # attributeSelected = attribute
                    # attribute_extractedSelected = attribute_extracted
                else:
                    predAllar = np.concatenate((predAllar,predar),axis = 0)
                    predAllrf = np.concatenate((predAllrf, predrf), axis=0)
                    obsAll = np.concatenate((obsAll,obs),axis = 0)
                    # attributeSelected = np.concatenate((attributeSelected, attribute), axis=0)
                    # attribute_extractedSelected = np.concatenate((attribute_extractedSelected, attribute_extracted), axis=0)
                count = count+1

                lat_selected.append(GAGEII_lat[gageIdx])
                lon_selected.append(GAGEII_lon[gageIdx])
                meanflowrate_selected.append(np.nanmean(obs))
                GageII_selected.append(gageID)
    except:
        print(basinID, "is not selected now")






import scipy.stats

def statError(pred, target):
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)

    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
    # ubRMSE
    #dMax_rel = (np.nanmax(pred,axis=1)-np.nanmax(target,axis=1))/np.nanmax(target,axis=1)
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
    # FDC metric
    # predFDC = calFDC(pred)
    # targetFDC = calFDC(target)
    #FDCRMSE = np.sqrt(np.nanmean((predFDC - targetFDC) ** 2, axis=1))
    # rho R2 NSE
    dMax_rel = np.full(ngrid, np.nan)
    dMax = np.full(ngrid, np.nan)
    Corr = np.full(ngrid, np.nan)
    Bias_rel = np.full(ngrid, np.nan)
    CorrSp = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    NNSE = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    PBiasother = np.full(ngrid, np.nan)
    absPBiaslow = np.full(ngrid, np.nan)
    absPBiashigh = np.full(ngrid, np.nan)
    absPBias = np.full(ngrid, np.nan)
    absPBiasother = np.full(ngrid, np.nan)
    KGE = np.full(ngrid, np.nan)
    KGE12 = np.full(ngrid, np.nan)
    RMSElow = np.full(ngrid, np.nan)
    RMSEhigh = np.full(ngrid, np.nan)
    RMSEother = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]


        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]


        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            maxobs = np.nanmax(yy)
            maxIdx = np.nanargmax(yy)
            window_lower = 10
            window_upper = 11
            if (maxIdx < window_lower):
                window_lower = maxIdx
            elif (window_upper > len(xx) - maxIdx):
                window_upper = len(xx) - maxIdx

            maxpred = np.nanmax(xx[maxIdx - window_lower:maxIdx + window_upper])
            #  maxpred = np.nanmax(x)
            dMax[k] = maxpred - maxobs
            dMax_rel[k] = (maxpred - maxobs) / maxobs * 100


            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100

            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.98 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            otherpred = pred_sort[indexlow:indexhigh]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            othertarget = target_sort[indexlow:indexhigh]
            PBiaslow[k] = np.sum((lowpred - lowtarget)) / (np.sum(lowtarget) +0.0001)* 100
            PBiashigh[k] = np.sum((highpred - hightarget) )/ np.sum(hightarget) * 100
            PBiasother[k] = np.sum((otherpred - othertarget)) / np.sum(othertarget) * 100
            absPBiaslow[k] = np.sum(abs(lowpred - lowtarget)) / ((np.sum(lowtarget) +0.0001))* 100
            absPBiashigh[k] = np.sum(abs(highpred - hightarget) )/ np.sum(hightarget) * 100
            absPBiasother[k] = np.sum(abs(otherpred - othertarget)) / np.sum(othertarget) * 100
            Bias_rel[k] = (np.sum(xx)-np.sum(yy))/np.sum(yy)

            RMSElow[k] = np.sqrt(np.nanmean((lowpred - lowtarget)**2))
            RMSEhigh[k] = np.sqrt(np.nanmean((highpred - hightarget)**2))
            RMSEother[k] = np.sqrt(np.nanmean((otherpred - othertarget)**2))

            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                CorrSp[k] = scipy.stats.spearmanr(xx, yy)[0]
                yymean = yy.mean()
                yystd = np.std(yy)
                xxmean = xx.mean()
                xxstd = np.std(xx)
                KGE[k] = 1 - np.sqrt((Corr[k]-1)**2 + (xxstd/yystd-1)**2 + (xxmean/yymean-1)**2)
                KGE12[k] = 1 - np.sqrt((Corr[k] - 1) ** 2 + ((xxstd*yymean)/ (yystd*xxmean) - 1) ** 2 + (xxmean / yymean - 1) ** 2)
                SST = np.sum((yy-yymean)**2)
                SSReg = np.sum((xx-yymean)**2)
                SSRes = np.sum((yy-xx)**2)
                R2[k] = 1-SSRes/SST
                NSE[k] = 1-SSRes/SST
                NNSE[k] = SST/(SSRes+SST)

    outDict = dict(Bias=Bias,Bias_rel=Bias_rel, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, CorrSp=CorrSp, R2=R2, NSE=NSE,NNSE=NNSE,
                   FLV=PBiaslow, FHV=PBiashigh, PBias=PBias, PBiasother=PBiasother, absFLV=absPBiaslow, absFHV=absPBiashigh, absPBias=absPBias, absPBiasother=absPBiasother, KGE=KGE, KGE12=KGE12,
                   lowRMSE=RMSElow, highRMSE=RMSEhigh, midRMSE=RMSEother,rdMax = dMax_rel,dMax = dMax)
    return outDict






evaDict = [statError(predAllar[:,:], obsAll[:,:])]
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


print("Routed flow model'NSE', 'Bias','Corr','Bias_rel",'NNSE',
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]), np.nanmedian(dataBox[4][0]))

NNSElst_ar =dataBox[4][0]
correlationslst_ar  =dataBox[2][0]



evaDict = [statError(predAllrf[:,:], obsAll[:,:])]
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


print("Forward flow model'NSE', 'Bias','Corr','Bias_rel",'NNSE',
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]), np.nanmedian(dataBox[4][0]))

NNSElst_rf =dataBox[4][0]
correlationslst_rf  =dataBox[2][0]

idx_ar = np.where((NNSElst_ar - NNSElst_rf)>0.1 )[0]

for idx_ in range(len(idx_ar)):

    # Plot

    idx = idx_ar[idx_]
    if area_selected[idx] >800:
       # idx = idx_ar[23]
        fontsize = 18
        plt.rcParams.update({'font.size': fontsize})
        plotTime = pd.date_range('1985-10-05', f'1995-09-25', freq='d')

        plt.figure(figsize=(10, 6))
        plt.plot(routedTime[routedTime.get_loc(plotTime[0]):routedTime.get_loc(plotTime[-1])], predAllar[idx,routedTime.get_loc(plotTime[0]):routedTime.get_loc(plotTime[-1])], label=f'Routed flow: NSE {round(NNSElst_ar[idx],2)}',lw = 2, color='blue')
        plt.plot(routedTime[routedTime.get_loc(plotTime[0]):routedTime.get_loc(plotTime[-1])], predAllrf[idx, routedTime.get_loc(plotTime[0]):routedTime.get_loc(plotTime[-1])], label=f'Forward runoff: NSE {round(NNSElst_rf[idx],2)}',lw = 2, color='red')
        plt.plot(routedTime[routedTime.get_loc(plotTime[0]):routedTime.get_loc(plotTime[-1])], obsAll[idx,routedTime.get_loc(plotTime[0]):routedTime.get_loc(plotTime[-1])], label='Observation',lw = 2, color='k')
        plt.title(f'Gage {GageII_selected[idx]}, Catchment area {area_selected[idx]} km$^2$')
        plt.xlabel('Date')
        plt.ylabel(r'Discharge (m$^3$/s)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=True)

# # Example data: Replace with your actual streamflow prediction and observation data.
# predictions = np.array([...])  # Streamflow predictions
# observations = np.array([...])  # Streamflow observations

# Calculate the bias percentage

#
# fontsize = 18
# plt.rcParams.update({'font.size': fontsize})
# fig, ax = plt.subplots(figsize=(11, 6))  # Example size: 10 inches by 6 inches
#
#
# # Define the bins for categorization
# # Define the bins for categorization
# bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#
# # Use numpy to categorize each correlation into bins and count them
# counts, _ = np.histogram(NNSElst_ar, bins)
#
# # Calculate the percentage of sites with correlation >= 0.8
# percentage_above_08 = (NNSElst_ar >= 0.6).sum() / len(NNSElst_ar) * 100
#
# # Bin labels for the plot
# bin_labels = ['(0,0.2)', '(0.2,0.4)', '(0.4,0.6)', '(0.6,0.8)', '(0.8,1)']
#
# # Create a bar chart
# colors = ['aliceblue', 'lightblue', 'mediumpurple', 'blueviolet','purple']  # Define your colors here
# plt.bar(bin_labels, counts, color=colors)
#
# # Create a legend
# colors = {'(0,0.2]':'aliceblue', '(0.2,0.4]':'lightblue', '(0.4,0.6]':'mediumpurple', '(0.6,0.8]':'blueviolet', '(0.8,1]':'purple'}
# labels = list(colors.keys())
# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# plt.legend(handles, labels, title="NNSE", loc='upper left', bbox_to_anchor=(1.05, 0.8))
#
# # Add title and labels
# #plt.title(r'Distribution of NNSE ($\delta$ HBV model)')
# plt.xlabel('NNSE')
# plt.ylabel('Site Count')
# plt.ylim([0,450])
#
# # Adding the annotation for the percentage of sites with cor >= 0.8
#
# plt.text(0.2, -100, f'{percentage_above_08:.0f}% have NNSE >= 0.6',
#           color='brown', fontweight='bold',clip_on=False,fontsize = 24)
# plt.tight_layout()
# plt.savefig(data_folder + "routedNNSE.png", dpi=300)
# #plt.savefig(data_folder + "RunoffNNSE.png", dpi=300)
# # Display the plot
# plt.show(block=True)
#
#
#
# fontsize = 18
# plt.rcParams.update({'font.size': fontsize})
# fig, ax = plt.subplots(figsize=(11, 6))  # Example size: 10 inches by 6 inches
#
#
# # Define the bins for categorization
# # Define the bins for categorization
# bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#
# # Use numpy to categorize each correlation into bins and count them
# counts, _ = np.histogram(NNSElst_rf, bins)
#
# # Calculate the percentage of sites with correlation >= 0.8
# percentage_above_08 = (NNSElst_rf >= 0.6).sum() / len(NNSElst_rf) * 100
#
# # Bin labels for the plot
# bin_labels = ['(0,0.2)', '(0.2,0.4)', '(0.4,0.6)', '(0.6,0.8)', '(0.8,1)']
#
# # Create a bar chart
# colors = ['aliceblue', 'lightblue', 'mediumpurple', 'blueviolet','purple']  # Define your colors here
# plt.bar(bin_labels, counts, color=colors)
#
# # Create a legend
# colors = {'(0,0.2]':'aliceblue', '(0.2,0.4]':'lightblue', '(0.4,0.6]':'mediumpurple', '(0.6,0.8]':'blueviolet', '(0.8,1]':'purple'}
# labels = list(colors.keys())
# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# plt.legend(handles, labels, title="NNSE", loc='upper left', bbox_to_anchor=(1.05, 0.8))
#
# # Add title and labels
# #plt.title(r'Distribution of NNSE ($\delta$ HBV model)')
# plt.xlabel('NNSE')
# plt.ylabel('Site Count')
# plt.ylim([0,450])
#
# # Adding the annotation for the percentage of sites with cor >= 0.8
#
# plt.text(0.2, -100, f'{percentage_above_08:.0f}% have NNSE >= 0.6',
#           color='brown', fontweight='bold',clip_on=False,fontsize = 24)
# plt.tight_layout()
# #plt.savefig(data_folder + "routedNNSE.png", dpi=300)
# plt.savefig(data_folder + "RunoffNNSE.png", dpi=300)
# # Display the plot
# plt.show(block=True)
#
#
#
# #correlations_org = dataBox[2][0]
# #correlations = correlations_org[np.where(correlations_org==correlations_org)]
# fontsize = 18
# plt.rcParams.update({'font.size': fontsize})
# fig, ax = plt.subplots(figsize=(11, 6))  # Example size: 10 inches by 6 inches
#
#
# # Define the bins for categorization
# # Define the bins for categorization
# bins = [-1.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#
# # Use numpy to categorize each correlation into bins and count them
# counts, _ = np.histogram(correlationslst_ar, bins)
#
# # Calculate the percentage of sites with correlation >= 0.8
# percentage_above_08 = (correlationslst_ar >= 0.8).sum() / len(correlationslst_ar) * 100
#
# # Bin labels for the plot
# bin_labels = ['(-1,0.2)', '(0.2,0.4)', '(0.4,0.6)', '(0.6,0.8)', '(0.8,1)']
#
# # Create a bar chart
# colors = ['aliceblue', 'lightblue', 'mediumpurple', 'blueviolet','purple']  # Define your colors here
# plt.bar(bin_labels, counts, color=colors)
#
# # Create a legend
# colors = {'(-1,0.2]':'aliceblue', '(0.2,0.4]':'lightblue', '(0.4,0.6]':'mediumpurple', '(0.6,0.8]':'blueviolet', '(0.8,1]':'purple'}
# labels = list(colors.keys())
# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# plt.legend(handles, labels, title="Correlation", loc='upper left', bbox_to_anchor=(1.05, 0.8))
#
# # Add title and labels
# #plt.title(r'Distribution of Correlations ($\delta model$)')
# plt.xlabel('Correlations')
# plt.ylabel('Site Count')
# plt.ylim([0,600])
# # Adding the annotation for the percentage of sites with cor >= 0.8
# plt.text(-0.3, -130, f'{percentage_above_08:.0f}% have Correlations >= 0.6',
#           color='brown', fontweight='bold',clip_on=False,fontsize = 24)
# # plt.text(1, 100, f'{percentage_above_08:.0f}% have Correlations >= 0.8',
# #          horizontalalignment='center', color='black', fontweight='bold',clip_on=False)
# plt.tight_layout()
# plt.savefig(data_folder + "routedCorrelations.png", dpi=300)
# #plt.savefig(data_folder + "RunoffCorrelations.png", dpi=300)
# # Display the plot
# plt.show(block=True)
#
# fontsize = 18
# plt.rcParams.update({'font.size': fontsize})
# fig, ax = plt.subplots(figsize=(11, 6))  # Example size: 10 inches by 6 inches
#
#
# # Define the bins for categorization
# # Define the bins for categorization
# bins = [-1.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#
# # Use numpy to categorize each correlation into bins and count them
# counts, _ = np.histogram(correlationslst_rf, bins)
#
# # Calculate the percentage of sites with correlation >= 0.8
# percentage_above_08 = (correlationslst_rf >= 0.8).sum() / len(correlationslst_rf) * 100
#
# # Bin labels for the plot
# bin_labels = ['(-1,0.2)', '(0.2,0.4)', '(0.4,0.6)', '(0.6,0.8)', '(0.8,1)']
#
# # Create a bar chart
# colors = ['aliceblue', 'lightblue', 'mediumpurple', 'blueviolet','purple']  # Define your colors here
# plt.bar(bin_labels, counts, color=colors)
#
# # Create a legend
# colors = {'(-1,0.2]':'aliceblue', '(0.2,0.4]':'lightblue', '(0.4,0.6]':'mediumpurple', '(0.6,0.8]':'blueviolet', '(0.8,1]':'purple'}
# labels = list(colors.keys())
# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# plt.legend(handles, labels, title="Correlation", loc='upper left', bbox_to_anchor=(1.05, 0.8))
#
# # Add title and labels
# #plt.title(r'Distribution of Correlations ($\delta model$)')
# plt.xlabel('Correlations')
# plt.ylabel('Site Count')
# plt.ylim([0,600])
# # Adding the annotation for the percentage of sites with cor >= 0.8
# plt.text(-0.3, -130, f'{percentage_above_08:.0f}% have Correlations >= 0.6',
#           color='brown', fontweight='bold',clip_on=False,fontsize = 24)
# # plt.text(1, 100, f'{percentage_above_08:.0f}% have Correlations >= 0.8',
# #          horizontalalignment='center', color='black', fontweight='bold',clip_on=False)
# plt.tight_layout()
# #plt.savefig(data_folder + "routedCorrelations.png", dpi=300)
# plt.savefig(data_folder + "RunoffCorrelations.png", dpi=300)
# # Display the plot
# plt.show(block=True)
#
#
# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.patches import Rectangle
# nbin = 5
# lower_bound = 0
# upper_bound = 800
# #bins = np.linspace(lower_bound, upper_bound, nbin + 1)
# bin_length = (upper_bound - lower_bound) / (nbin-1)
# bins =np.array([0,200,400,600,800,1000])
# # lat_bin_index = np.digitize(lat, bins)
# # #elev_bin_index = np.digitize(mean_elev, bins)
# area_selected =np.array(area_selected)
# area_bin_index = np.digitize(area_selected, bins)
# plt.rcParams.update({'font.size': 22})
# fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
# labels = []
# for bin_i in range(len(bins)-1):
#     labels.append(f'{bins[bin_i]}~{bins[bin_i+1]}')
#
# plot1 = ax.boxplot( [ NNSElst_rf[np.where(area_bin_index == i)][~np.isnan(NNSElst_rf[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+1*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="aliceblue", color="k") ,widths = bin_length/4)
# plot2 = ax.boxplot( [ NNSElst_ar[np.where(area_bin_index == i)][~np.isnan(NNSElst_ar[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+3*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="mediumpurple", color="k"),widths = bin_length/4 )
#
# for whisker in plot1['whiskers']:
#     whisker.set(ls='-', linewidth=2,color = "k")
# for cap in plot1['caps']:
#     cap.set(ls='-', linewidth=2,color = "k")
# for box in plot1['boxes']:
#     box.set(ls='-', linewidth=2)
# for median in plot1['medians']:
#     median.set(ls='-', linewidth=2,color = "k")
# for whisker in plot2['whiskers']:
#     whisker.set(ls='-', linewidth=2,color = "k")
# for cap in plot2['caps']:
#     cap.set(ls='-', linewidth=2,color = "k")
# for box in plot2['boxes']:
#     box.set(ls='-', linewidth=2)
# for median in plot2['medians']:
#     median.set(ls='-', linewidth=2,color = "k")
#
# for i in range(1,nbin+1):
#     #num_local = len(localnse_all[np.where(drainage_area_bin_index == i)])
#     #num_whole = len(wholense_all[np.where(drainage_area_bin_index == i)])
#     num = len(area_selected[np.where(area_bin_index == i)])
#     ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,1.15, f'{num} sites')
#
# ax.add_patch( Rectangle(( 700, 0.25),80, 0.07,  fc = "aliceblue",  ec ='k',ls = "-" , lw = 2) )
# ax.text(800, 0.25, r"Runoff")
# ax.add_patch( Rectangle(( 700, 0.1), 80, 0.07,  fc = "mediumpurple",  ec ='k',ls = "--" , lw = 2) )
# ax.text(800, 0.1, r"Routed flow")
# ax.set_ylabel("$NNSE$")
# ax.set_xlabel(r"Drainage area (km$^2$)")
#
# ax.set_yticks(np.arange(0,1.01,0.4))
# ax.set_ylim([0,1.31])
# ax.set_xlim([lower_bound,upper_bound+bin_length])
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# # ax.vlines([-0.5,0,0.5], -2, 4,color ="k",linestyles='--',lw = 1.5)
# ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), -2,1.5,color ="k",linestyles='--',lw = 2.5)
# tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
# ax.set_xticks(tick_positions)
# #ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
# ax.set_xticklabels(labels)
#
# plt.savefig(data_folder + "boxplot_NSE_area.png", dpi=300)
# plt.show(block=True)
#
# lat_selected = np.array(lat_selected)[:,0]
# lon_selected = np.array(lon_selected)[:,0]
# fontsize = 12
# plt.rcParams.update({'font.size': fontsize})
#
#
# fig, ax = plt.subplots(figsize=(8, 6))
# bounding = [np.min(lat_selected)-0.5, np.max(lat_selected)+0.5,
#                     np.min(lon_selected)-0.5,np.max(lon_selected)+0.5]
# prj='cyl'
# mm = basemap.Basemap(
#     llcrnrlat=bounding[0],
#     urcrnrlat=bounding[1],
#     llcrnrlon=bounding[2],
#     urcrnrlon=bounding[3],
#     projection=prj,
#     resolution='c',
#     ax=ax)
# mm.drawcoastlines()
# mm.drawstates(linestyle='dashed')
# mm.drawcountries(linewidth=1.0, linestyle='-.')
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# x, y = mm(lon_selected, lat_selected)
# dNNSE = NNSElst_ar-NNSElst_rf
#
# # Normalize the colors based on the range of dNNSE
# norm = Normalize(vmin=min(dNNSE), vmax=max(dNNSE))
#
# # Create a ScalarMappable object with the normalization and the colormap
# mappable = ScalarMappable(norm=norm, cmap='jet')
# mappable.set_array(dNNSE)
# # Now plot each point individually (simulating the existing loop)
# for i in range(len(x)):
#     mm.scatter(x[i:i + 1], y[i:i + 1], c=dNNSE[i:i + 1], marker='o', s=area_selected[i]**1.5/200, cmap='jet', norm=norm)
#
#
#
# cbar = fig.colorbar(mappable, ax=ax, orientation='vertical', pad=0.01, fraction=0.019)
# cbar.set_label(r'$\Delta$ NSE')
# plt.savefig(data_folder + "Map_NSE_area.png", dpi=300)
# plt.show()


