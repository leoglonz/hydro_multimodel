import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import pandas as pd

import json
import glob
import sys
sys.path.append('../../')
from hydroDL.post import plot, stat

from mpl_toolkits import basemap


def _basin_norm_for_LSTM(
    x: np.array, basin_area: np.array, mean_prep: np.array, to_norm: bool
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
    tempprep = np.tile(mean_prep, (1, x.shape[1]))
    if to_norm is True:
        flow = (x * 0.0283168 * 3600 * 24) / (
            (temparea * (10**6)) * (tempprep * 10 ** (-3))
        )  # (m^3/day)/(m^3/day)

        #flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

    else:
        flow = (
            x
            * ((temparea * (10**6)) * (tempprep * 10 ** (-3)))
            / (0.0283168 * 3600 * 24)
        )
        # flow = (
        #     x
        #     * ((temparea * (10**6)) * (10 ** (-3)))
        #     / (0.0283168 * 3600 * 24)
        # )

    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow

def _basin_norm(
    x: np.array, basin_area: np.array, mean_prep: np.array, to_norm: bool
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
    tempprep = np.tile(mean_prep, (1, x.shape[1]))
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




def _trans_norm(
    x: np.array, var_lst: list, stat_dict: dict, log_norm_cols: list, to_norm: bool
) -> np.array:
    """
    Normalization or inverse normalization

    There are two normalization formulas:

    .. math:: normalized_x = (x - mean) / std

    and

     .. math:: normalized_x = [log_{10}(\sqrt{x} + 0.1) - mean] / std

     The later is only for vars in log_norm_cols; mean is mean value; std means standard deviation

    Parameters
    ----------
    x
        data to be normalized or denormalized
    var_lst
        the type of variables
    stat_dict
        statistics of all variables
    log_norm_cols
        which cols use the second norm method
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.full(x.shape, np.nan)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                if var in log_norm_cols:
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var in log_norm_cols:
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var in log_norm_cols:
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var in log_norm_cols:
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
    return out






data_folder_3200 = "/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/gages/dataCONUS3200/"


with open(data_folder_3200+'train_data_dict.json') as f:
    train_data_dict = json.load(f)
smallBasinID = train_data_dict['sites_id']

with open('/data/yxs275/DPL_HBV/CONUS_3200_Output/LSTM_local_daymet_filled_withNaN_NSE/exp_EPOCH300_BS100_RHO365_HS256_trainBuff365/' + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)

data_folder = "/data/yxs275/CONUS_data/HUC10/version_1_11_2014_continental_routing/"
#data_split_folder = "/data/yxs275/CONUS_data/HUC10/dPL_1_11_2024/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/basin_split/"

data_split_folder = "/data/yxs275/CONUS_data/HUC10/LSTM_local_daymet_filled_withNaN_NSE/basin_split/"
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
GAGEII_pmean = GAGEII_attr[:,np.where(np.array(attributeAllLst)=="PPTAVG_BASIN")[0]]/365*10
GAGEII_lat = GAGEII_attr[:,np.where(np.array(attributeAllLst)=="LAT_GAGE")[0]]
GAGEII_lon = GAGEII_attr[:,np.where(np.array(attributeAllLst)=="LNG_GAGE")[0]]
streamflow_trans = _basin_norm(
                        GAGEII_flow[:, :, 0 :  1].copy(), GAGEII_area,GAGEII_pmean, to_norm=True
                    )


data_folder_new = "/data/yxs275/CONUS_data/HUC10/version_1_11_2014_continental_routing/"
selected_Basin = np.load(data_folder_new+"selected_Basin.npy")
selected_GAGE = np.load(data_folder_new+"selected_GAGE.npy")

Insmall_list = []
lat_selected = []
lon_selected = []
meanflowrate_selected = []
evaluateTime = pd.date_range('2013-10-01', f'2016-10-01', freq='d')
count = 0
Valid_data_percentage = []
HUC10_selected = []
GageII_selected = []
area_selected = []
pmean_selected = []
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
        pred = np.load(data_split_folder + f"{basinID}.npy")[:,
               AllHUC10Time.get_loc(evaluateTime[0]):AllHUC10Time.get_loc(evaluateTime[-1])]
        obs = streamflow_trans[gageIdx: gageIdx + 1, AllGageTime.get_loc(evaluateTime[0]):AllGageTime.get_loc(evaluateTime[-1]), 0]



        attribute = GAGEII_attr[gageIdx: gageIdx + 1,:] #attributeALL_df.values[basinIdx:basinIdx + 1, :]
        attribute_extracted =  attributeALL_df[attributeALL_df['gage_ID'] ==  int(basinID)].values[:,1:]
       # attribute = np.concatenate((np.expand_dims(gageID,axis = [0,1]), attribute),axis = 1)
        nan_count = np.isnan(obs).sum()
        print("Valid data percentage: ", nan_count/len(obs[0,:]))
        Valid_data_percentage.append(nan_count/len(obs[0,:]))
        if nan_count/len(obs[0,:]) <0.3:
            HUC10_selected.append(basinID)
            pred = _trans_norm(
                np.expand_dims(pred, axis=-1),
                ['usgsFlow'],
                stat_dict,
                log_norm_cols=[],
                to_norm=False,
            )[:, :, 0]

            if count == 0:

                predAll = pred
                obsAll = obs
                attributeSelected = attribute
                attribute_extractedSelected = attribute_extracted
            else:
                predAll = np.concatenate((predAll,pred),axis = 0)
                obsAll = np.concatenate((obsAll,obs),axis = 0)
                attributeSelected = np.concatenate((attributeSelected, attribute), axis=0)
                attribute_extractedSelected = np.concatenate((attribute_extractedSelected, attribute_extracted), axis=0)
            count = count+1
            area_selected.append(GAGEII_area[gageIdx])
            pmean_selected.append(GAGEII_pmean[gageIdx])
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




yPred = _basin_norm_for_LSTM(
    predAll.copy(), np.array(area_selected), np.array(pmean_selected), to_norm=False
)

yPred_mmday = _basin_norm(
    yPred.copy(), np.array(area_selected), np.array(pmean_selected), to_norm=True
)


evaDict = [statError(yPred_mmday[:,:], obsAll[:,:])]
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


print("LSTM model'NSE', 'Bias','Corr','Bias_rel",'NNSE',
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]), np.nanmedian(dataBox[4][0]))

NSElst =dataBox[0][0]

# for idx in range(len(NSElst)):
#
#     # Plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(evaluateTime[:-1], predAll[idx,:], label='Prediction', color='blue')
#     plt.plot(evaluateTime[:-1], obsAll[idx,:], label='Observation', color='green')
#     plt.title(f'Basin {HUC10_selected[idx]} Streamflow Prediction vs. Observation NNSE is {NSElst[idx]}')
#     plt.xlabel('Date')
#     plt.ylabel('Streamflow')
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show(block=True)

# # Example data: Replace with your actual streamflow prediction and observation data.
# predictions = np.array([...])  # Streamflow predictions
# observations = np.array([...])  # Streamflow observations

# Calculate the bias percentage
NNSE = dataBox[4][0]
NNSE = NNSE[np.where(NNSE==NNSE)]

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
plt.title(r'Distribution of NNSE ($\delta$ HBV model)')
plt.xlabel('NNSE')
plt.ylabel('Site Count')
plt.tight_layout()
# Adding the annotation for the percentage of sites with cor >= 0.8

# plt.text(0.7, 100, f'{percentage_above_08:.0f}% have NNSE >= 0.6',
#          horizontalalignment='center', color='black', fontweight='bold',clip_on=False)

#plt.savefig(data_folder + "NNSE.png", dpi=300)
# Display the plot
plt.show(block=True)



correlations_org = dataBox[2][0]
correlations = correlations_org[np.where(correlations_org==correlations_org)]
fontsize = 18
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(11, 6))  # Example size: 10 inches by 6 inches


# Define the bins for categorization
# Define the bins for categorization
bins = [-1.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Use numpy to categorize each correlation into bins and count them
counts, _ = np.histogram(correlations, bins)

# Calculate the percentage of sites with correlation >= 0.8
percentage_above_08 = (correlations >= 0.8).sum() / len(correlations) * 100

# Bin labels for the plot
bin_labels = ['(-1,0.2)', '(0.2,0.4)', '(0.4,0.6)', '(0.6,0.8)', '(0.8,1)']

# Create a bar chart
colors = ['aliceblue', 'lightblue', 'mediumpurple', 'blueviolet','purple']  # Define your colors here
plt.bar(bin_labels, counts, color=colors)

# Create a legend
colors = {'(-1,0.2]':'aliceblue', '(0.2,0.4]':'lightblue', '(0.4,0.6]':'mediumpurple', '(0.6,0.8]':'blueviolet', '(0.8,1]':'purple'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, title="Correlation", loc='upper left', bbox_to_anchor=(1.05, 0.8))

# Add title and labels
plt.title(r'Distribution of Correlations ($\delta model$)')
plt.xlabel('Correlations')
plt.ylabel('Site Count')

# Adding the annotation for the percentage of sites with cor >= 0.8

# plt.text(1, 100, f'{percentage_above_08:.0f}% have Correlations >= 0.8',
#          horizontalalignment='center', color='black', fontweight='bold',clip_on=False)
plt.tight_layout()
plt.savefig(data_folder + "Correlations.png", dpi=300)
# Display the plot
plt.show(block=True)



lat_selected = np.array(lat_selected)[np.where(correlations_org==correlations_org)]
lon_selected = np.array(lon_selected)[np.where(correlations_org==correlations_org)]
meanflowrate_selected = np.array(meanflowrate_selected)[np.where(correlations_org==correlations_org)]
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# Load your dataset here, which must include latitude, longitude, correlation, and mean flowrate.

# Create a Basemap instance
m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50,
            llcrnrlon=-125, urcrnrlon=-65, resolution='i')

m.drawcoastlines()
m.drawcountries()
m.drawstates()


# Convert latitude and longitude to x and y coordinates
x, y = m(lon_selected, lat_selected)

# Plot each point, with size and color determined by mean flowrate and correlation
for i in range(len(x)):
    m.scatter(x[i], y[i], s=meanflowrate_selected[i]/10, c=correlations[i], cmap='Purple', alpha=0.5)

# Create a colorbar and a legend
plt.colorbar(label='Correlation')
plt.clim(0, 1)  # Assuming correlation values are between 0 and 1

# Assuming that mean_flowrate could be used directly as sizes for plotting
sizes = [50, 100, 150, 200]
for size in sizes:
    plt.scatter([], [], s=size, c='k', alpha=0.5, label=str(size) + ' cms')

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Mean Flowrate (cms)')

plt.title('Map of NWM v2.1 Streamflow Correlation')
plt.show()
