import sys

from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))


from hydroDL.model import rnn, crit, train
from hydroDL.data import camels
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
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _basin_norm_to_m3_s(
    x: np.array, basin_area: np.array, to_norm: bool
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
    if x.shape[0] != basin_area.shape[0]:
        basin_area = np.expand_dims(basin_area,axis = 0)
        basin_area = np.tile(basin_area,(x.shape[0],1))
    temparea = np.tile(basin_area, (1, x.shape[1]))

    if to_norm is True:


        flow = (x  * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

    else:

        flow = (
            x
            * ((temparea * (10**6)) * (10 ** (-3)))
            / ( 3600 * 24)
        )

    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow



def _basin_norm(
    x: np.array, basin_area: np.array, to_norm: bool
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


def trans_norm(x, var_lst, stat_dict, *, to_norm):
    """
    normalization, including denormalization code

    Parameters
    ----------
    x
        2d or 3d data
        2d：1st-sites，2nd-var type
        3d：1st-sites，2nd-time, 3rd-var type
    var_lst
        variables
    stat_dict
        a dict with statistics info
    to_norm
        if True, normalization; else denormalization

    Returns
    -------
    np.array
        normalized/denormalized data
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.zeros(x.shape)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
    return out

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


def hargreaves(tmin, tmax, tmean, lat, trange):
    # calculate the day of year
    dfdate = pd.date_range(start=str(trange[0]), end=str(trange[1]), freq='D')[:-1] # end not included
    tempday = np.array(dfdate.dayofyear)
    day_of_year = np.tile(tempday.reshape(-1, 1), [1, tmin.shape[-1]])
    # Loop to reduce memory usage
    pet = np.zeros(tmin.shape, dtype=np.float32) * np.NaN
    for ii in np.arange(len(pet[:, 0])):
        trange = tmax[ii, :] - tmin[ii, :]
        trange[trange < 0] = 0

        latitude = np.deg2rad(lat[ii, :])

        SOLAR_CONSTANT = 0.0820

        sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * day_of_year[ii, :] - 1.39))

        sha = np.arccos(np.clip(-np.tan(latitude) * np.tan(sol_dec), -1, 1))

        ird = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year[ii, :]))

        tmp1 = (24.0 * 60.0) / np.pi
        tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
        tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
        et_rad = tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)

        pet[ii, :] = 0.0023 * (tmean[ii, :] + 17.8) * trange ** 0.5 * 0.408 * et_rad

    pet[pet < 0] = 0

    return pet

def cal_4_stat_inds(b):
    """
    Calculate four statistics indices: percentile 10 and 90, mean value, standard deviation

    Parameters
    ----------
    b
        input data

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def cal_stat_gamma(x):
    """
    Try to transform a time series data to normal distribution

    Now only for daily streamflow, precipitation and evapotranspiration;
    When nan values exist, just ignore them.

    Parameters
    ----------
    x
        time series data

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    return cal_4_stat_inds(b)

def cal_stat(x: np.array) -> list:
    """
    Get statistic values of x (Exclude the NaN values)

    Parameters
    ----------
    x: the array

    Returns
    -------
    list
        [10% quantile, 90% quantile, mean, std]
    """
    a = x.flatten()
    b = a[~np.isnan(a)]
    if b.size == 0:
        # if b is [], then give it a 0 value
        b = np.array([0])
    return cal_4_stat_inds(b)

def fill_Nan(array_3d):
    # Define the x-axis for interpolation
    x = np.arange(array_3d.shape[1])

    # Iterate over the first and third dimensions to interpolate the second dimension
    for i in range(array_3d.shape[0]):
        for j in range(array_3d.shape[2]):
            # Select the 1D slice for interpolation
            slice_1d = array_3d[i, :, j]

            # Find indices of NaNs and non-NaNs
            nans = np.isnan(slice_1d)
            non_nans = ~nans

            # Only interpolate if there are NaNs and at least two non-NaN values for reference
            if np.any(nans) and np.sum(non_nans) > 1:
                # Perform linear interpolation using numpy.interp
                array_3d[i, :, j] = np.interp(x, x[non_nans], slice_1d[non_nans], left=None, right=None)
    return array_3d
## GPU setting
# which GPU to use when having multiple
traingpuid = 6
torch.cuda.set_device(traingpuid)


data_folder = "/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/gages/dataCONUS3200/"
#data_folder = "/storage/work/yxs275/DPL_HBV/CONUS_3200_data/dataCONUS3200/"
# data_folder_selected = "/data/yxs275/CONUS_data/HUC10/version_2_11_25/"
# selected_gageIdx = np.load(data_folder_selected+"selected_gageIdx.npy")


with open(data_folder+'test_data_dict.json') as f:
    train_data_dict = json.load(f)

# forcingAll_test = np.load(data_folder+"test_forcing.npy")
streamflow_test = np.load(data_folder+"test_flow.npy")
# forcingAll_train = np.load(data_folder+"train_forcing.npy")
streamflow_train = np.load(data_folder+"train_flow.npy")
warmup_span = pd.date_range('1992-10-01',f'1995-09-30', freq='d')
Tex = [19921001, 20101001]
warmup_for_testing = len(warmup_span)

AllTime = pd.date_range('1980-01-01', f'2020-12-31', freq='d')
# index_start_tadd = AllTime.get_loc(warmup_span[0])
# index_end_tadd = AllTime.get_loc('2010-09-30')

index_start = AllTime.get_loc(warmup_span[0])
index_end = AllTime.get_loc('2010-09-30')

forcingAll = np.load("/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/allBasin_localDaymet.npy")[:,index_start:index_end+1,:]
shapeID_str_lst= np.load("/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/shapeID_str_lst.npy")

forcingAll = fill_Nan(forcingAll)
# forcingAll = np.concatenate((forcingAll_train[:,-warmup_for_testing:,:],forcingAll_test),axis = 1)
streamflow = np.concatenate((streamflow_train[:,-warmup_for_testing:,:],streamflow_test),axis = 1)

# forcingAllLst  = train_data_dict['relevant_cols']
forcingAllLst  = ['prcp', 'tmax', 'tmin']

attribute_file = '/projects/mhpi/yxs275/Data/attributes_haoyu/attributes_haoyu.csv'
attributeALL_df = pd.read_csv(attribute_file,index_col=0)
attributeALL_df = attributeALL_df.sort_values(by='id')

attributeAllLst = attributeALL_df.columns

basin_area = np.expand_dims(attributeALL_df["area"].values,axis = 1)
lat =  attributeALL_df["lat"].values
idLst_new = attributeALL_df["id"].values
idLst_old = [int(id) for id in shapeID_str_lst]
[C, ind1, SubInd_id] = np.intersect1d(idLst_new, idLst_old, return_indices=True)
forcingAll = forcingAll[SubInd_id,:,:]
streamflow = streamflow[SubInd_id,:,:]


log_norm_cols=['prcp', 'pr', 'total_precipitation', 'pre', 'potential_evaporation', 'LE',
                   'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'ET_water', 'ET_sum', 'susm', 'smp', 'ssma', 'susma',
                   'usgsFlow', 'streamflow', 'qobs']
attributeLst = ['area','ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']
forcing_HBV_List = ['prcp',  'tmax', 'tmin', ]



[C, _, SubInd] = np.intersect1d(attributeLst, attributeAllLst, return_indices=True)
attribute = attributeALL_df.iloc[ind1, np.sort(SubInd)].values

attributeLst  = list(attributeAllLst[np.sort(SubInd)])



tmin = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmin")[0][0]], 0,1)
tmax = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmax")[0][0]], 0,1)

tmean = (tmin+tmax)/2

latarray = np.tile(lat, [tmin.shape[0], 1])
pet = hargreaves(tmin, tmax, tmean, lat=latarray, trange=Tex)


xTrain = np.zeros([forcingAll.shape[0],forcingAll.shape[1],3])
xTrain[:,:,0] = forcingAll[:,:,np.where(np.array(forcingAllLst) == "prcp")[0][0]]
xTrain[:,:,1] = np.swapaxes(tmean, 0,1)
xTrain[:,:,2] = np.swapaxes(pet, 0,1)
# xTrain[xTrain!=xTrain]  = 0


streamflow_trans = _basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


rootOut = "/projects/mhpi/yxs275/model/" + '/dPL_local_daymet_new_attr/'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, "exp_EPOCH50_BS100_RHO365_HS512_trainBuff365")  # output folder to save results
#out = os.path.join(rootOut, "exp001_new")
if os.path.exists(out) is False:
    os.mkdir(out)


#with open(rootOut+'/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365dapengscaler_stat.json') as f:
with open(out + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)


forcing_LSTM_norm = _trans_norm(
    xTrain.copy(), ['prcp','tmean','pet'], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)
xTrain[xTrain!=xTrain]  = 0
forcing_LSTM_norm[forcing_LSTM_norm!=forcing_LSTM_norm]  = 0

attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
attribute_norm[attribute_norm!=attribute_norm] = 0

attribute_norm_exp = np.expand_dims(attribute_norm, axis=1)
attribute_norm_exp = np.repeat(attribute_norm_exp, forcing_LSTM_norm.shape[1], axis=1)




testepoch = 50
model_path = out
print("Load model from ", model_path)
testmodel = loadModel(model_path, epoch=testepoch)

zTest = np.concatenate([forcing_LSTM_norm, attribute_norm_exp], 2)  # Add attributes to historical forcings as the inversion part
xTest = xTrain
testTuple = (xTest, zTest)
testbatch =300 #len(indexes)

A_change_list = [5,10,20,30,50,80,100,200,300,500,800,1000,1200,1500,1800,2000,3000,4000,5000,6000,8000,10000,15000,20000,30000,40000]
prediction_out = "/projects/mhpi/yxs275/DM_output/dPL_local_daymet_new_attr/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/"
#filePathLst = [prediction_out+"/allBasinParaHBV_base_new",prediction_out+"/allBasinParaRout_base_new",prediction_out+"/allBasinFluxes_base_new"]

filePathLst = [prediction_out + f"/allBasinParaHBV_area_new_{A_change_list[0]}", prediction_out + f"/allBasinParaRout_area_new_{A_change_list[0]}",prediction_out+f"/allBasinFluxes_area_new_{A_change_list[0]}"]


testmodel.inittime = 0
# allBasinFluxes_base,allBasinParaHBV_base,allBasinParaRout_base = train.visualParameters(
#     testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst,BufferLenth = warmup_for_testing)

allBasinParaHBV_base = np.load(filePathLst[0]+".npy")
allBasinParaRout_base = np.load(filePathLst[1]+".npy")
allBasinFluxes_base = np.load(filePathLst[2]+".npy")[:,-streamflow_test.shape[1]:,:]

dataPred = allBasinFluxes_base[:,:,0]
test_obs = streamflow_trans[:,warmup_for_testing:,0]
evaDict = [stat.statError(dataPred[:,:], test_obs)]
evaDictLst = evaDict
keyLst = ['NSE', 'Bias','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))


NSE_base = dataBox[0][0]
Bias_base = dataBox[1][0]




# evaDict = [stat.statError(dataPred[:,:warmup_for_testing], streamflow_trans[:,:warmup_for_testing,0])]
# evaDictLst = evaDict
# keyLst = ['NSE', 'Bias','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
# dataBox = list()
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     temp = list()
#     for k in range(len(evaDictLst)):
#         data = evaDictLst[k][statStr]
#         #data = data[~np.isnan(data)]
#         temp.append(data)
#     dataBox.append(temp)
# print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
#       np.nanmedian(dataBox[0][0]),
#       np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
#       np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))


# NSE_training_base = dataBox[0][0]
# Bias_training_base = dataBox[1][0]













time_allyear = pd.date_range(f'1995-10-01', f'2010-09-30', freq='d')
annual_flow_base= np.full((len(allBasinFluxes_base)),0)
annual_flow_obs= np.full((len(allBasinFluxes_base)),0)
for year in range(1995,2010):
    time_year = pd.date_range(f'{year}-10-01', f'{year+1}-09-30', freq='d')
    idx_start = time_allyear.get_loc(time_year[0])
    idx_end = time_allyear.get_loc(time_year[-1])
    annual_flow_base = annual_flow_base+np.sum(allBasinFluxes_base[:,idx_start:idx_end+1,0],axis = 1)
    annual_flow_obs = annual_flow_obs+np.sum(test_obs[:,idx_start:idx_end+1],axis = 1)









parameters = ["Beta","FC","K0","K1","K2","LP","PERC","UZL","TT","CFMAX","CFR","CWH","BETAET"]
selected_parameters = ["K0","K1","K2","Beta","FC","UZL"]

mean_area = np.nanmean(attribute[:, 0] )
std_area = np.nanstd(attribute[:, 0] )

#A_change_list = [10, 50, 100, 200, 500, 1000, 1500]
#A_change_list = [10, 50, 200, 500, 1000, 1500,5000,10000,50000]

#A_change_list = [0.1, 0.2, 0.5, 0.8, 1.0, 5,10,20,50,100,500]
# A_change_list = [0.001, 0.005, 0.01,0.05,0.1, 0.2, 0.5, 0.8, 1.0, 5,10,20,50,100,500]
mean_flow_frac_area = np.full((len(attribute),len(A_change_list)),np.nan)
Q2_Q_area = np.full((len(attribute),len(A_change_list)),np.nan)
peak_flow_area = np.full((len(attribute),len(A_change_list)),np.nan)
Bias_area = np.full((len(attribute),len(A_change_list)),np.nan)
HBVpara_area = np.full((len(attribute),len(A_change_list),len(selected_parameters)),np.nan)
Routpara_area = np.full((len(attribute),len(A_change_list),2),np.nan)

mean_flow_vol_area = np.full((len(attribute),len(A_change_list)),np.nan)
peak_flow_vol_area = np.full((len(attribute),len(A_change_list)),np.nan)

meanP = attributeALL_df["meanP"].values
area_idx=np.where(np.array(attributeLst) == "area")[0]

save_col = attribute[:, area_idx].copy()


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
for id_change in range(len(A_change_list)):

    
    attribute[:, area_idx] = A_change_list[id_change]  #*save_col
    attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
    attribute_norm[attribute_norm != attribute_norm] = 0

    attribute_norm_exp = np.expand_dims(attribute_norm, axis=1)
    attribute_norm_exp = np.repeat(attribute_norm_exp, forcing_LSTM_norm.shape[1], axis=1)

    zTest = np.concatenate([forcing_LSTM_norm, attribute_norm_exp],    2)  # Add attributes to historical forcings as the inversion part
    xTest = xTrain
    testTuple = (xTest, zTest)
    filePathLst = [prediction_out + f"/allBasinParaHBV_area_new_{A_change_list[id_change]}", prediction_out + f"/allBasinParaRout_area_new_{A_change_list[id_change]}",prediction_out+f"/allBasinFluxes_area_new_{A_change_list[id_change]}"]
    
    # allBasinFluxes,allBasinParaHBV, allBasinParaRout = train.visualParameters(
    #     testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst, BufferLenth=warmup_for_testing)
    allBasinParaHBV = np.load(filePathLst[0]+".npy")
    allBasinParaRout = np.load(filePathLst[1]+".npy")
    allBasinFluxes = np.load(filePathLst[2] + ".npy")[:,-streamflow_test.shape[1]:,:]
    



    Q2_Q = np.sum(allBasinFluxes[:,:,3],axis = 1)/(np.sum(allBasinFluxes[:,:,0],axis = 1)+0.00001)


    Bias,_ =bias_meanflowratio_calc(allBasinFluxes[:,:,0],test_obs) 

    mean_flow = np.full((len(allBasinFluxes)),0)
    peak_flow = np.full((len(allBasinFluxes)),0)

    mean_flow_vol = np.full((len(allBasinFluxes)),0)
    peak_flow_vol = np.full((len(allBasinFluxes)),0)

    for year in range(1995,2010):
        time_year = pd.date_range(f'{year}-10-01', f'{year+1}-09-30', freq='d')
        idx_start = time_allyear.get_loc(time_year[0])
        idx_end = time_allyear.get_loc(time_year[-1])

        peak_flow = peak_flow+np.max(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1)/(np.sum(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1)+0.00001)
        peak_flow_vol = peak_flow_vol + _basin_norm_to_m3_s(np.max(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1,keepdims=True),basin_area=np.array(A_change_list[id_change:id_change+1]),to_norm=False)[:,0]
    
    mean_flow_vol = mean_flow_vol + _basin_norm_to_m3_s(np.mean(allBasinFluxes[:,:,0],axis = 1,keepdims=True),basin_area=np.array(A_change_list[id_change:id_change+1]),to_norm=False)[:,0]
    _, mean_flow_ratio = bias_meanflowratio_calc(allBasinFluxes[:,:,0],allBasinFluxes_base[:,:,0]) 
    mean_flow = mean_flow+mean_flow_ratio

    peak_flow_frac = peak_flow/len(range(1995,2010))
    Q2_Q_area[:,id_change] = Q2_Q
    Bias_area[:,id_change] = Bias
    peak_flow_area[:,id_change] = peak_flow_frac
    mean_flow_frac_area[:,id_change] = mean_flow

    peak_flow_vol_area[:,id_change] = peak_flow_vol/len(range(1995,2010))
    mean_flow_vol_area[:,id_change] = mean_flow_vol

    [C, Ind, SubInd] = np.intersect1d(selected_parameters, parameters, return_indices=True)
    HBVpara_area[:,id_change,:] = allBasinParaHBV[:, np.sort(SubInd)]
    Routpara_area[:,id_change, :] = allBasinParaRout
    attribute[:,  area_idx] = save_col

lat =  attributeALL_df["lat"].values
lon=  attributeALL_df["lon"].values

sample_size = 500

aridity = attributeALL_df["aridity"].values

#level = [0.75,1,2]
level = [0.8,1.2,2.5]
very_arid = np.where(aridity>level[2])[0]
#very_arid = np.random.choice(very_arid_all, size=min(len(very_arid_all), sample_size), replace=False)

arid =np.where((aridity<=level[2]) & (aridity>level[1]))[0]
#arid = np.random.choice(arid_all, size=min(len(arid_all), sample_size), replace=False)
humid =np.where((aridity<=level[1]) & (aridity>level[0]))[0]
#humid = np.random.choice(humid_all, size=min(len(humid_all), sample_size), replace=False)
very_humid = np.where( (aridity<=level[0]))[0]
#very_humid = np.random.choice(very_humid_all, size=min(len(very_humid_all), sample_size), replace=False)
attribute_variable  = "permafrost"
ETPOT_Hargr = attributeALL_df[attribute_variable].values
ETPOT_Hargr_very_humid = ETPOT_Hargr[very_humid]
very_humid_with_low_ETPOT_Hargr = very_humid[np.where(ETPOT_Hargr_very_humid<=0.0001)[0]]

very_humid_with_middle_ETPOT_Hargr = very_humid[(np.where((ETPOT_Hargr_very_humid>0.0001 )  & (ETPOT_Hargr_very_humid<0.0007))[0])]

very_humid_with_high_ETPOT_Hargr = very_humid[np.where(ETPOT_Hargr_very_humid>=0.0007)[0]]

# basinsi_list = [very_arid,arid,humid,very_humid_with_low_ETPOT_Hargr,very_humid_with_middle_ETPOT_Hargr,very_humid_with_high_ETPOT_Hargr]
# climate_label = ['very_arid','arid','humid',f'very_humid_with_low_{attribute_variable}',f'very_humid_with_middle_{attribute_variable}',f'very_humid_with_high_{attribute_variable}']

basinsi_list = [very_arid,arid,humid,very_humid]
climate_label = ['very_arid','arid','humid','very_humid']


# meanslope = attributeALL_df["area"].values
# level = [200,1000,5000,10000]
# verylarge_slope_all = np.where(meanslope>level[2])[0]
# verylarge_slope = np.random.choice(verylarge_slope_all, size=min(len(verylarge_slope_all), sample_size), replace=False)

# large_slope_all =np.where((meanslope<=level[2]) & (meanslope>level[1]))[0]
# large_slope = np.random.choice(large_slope_all, size=min(len(large_slope_all), sample_size), replace=False)
# mild_slope_all =np.where((meanslope<=level[1]) & (meanslope>level[0]))[0]
# mild_slope = np.random.choice(mild_slope_all, size=min(len(mild_slope_all), sample_size), replace=False)
# verymild_slope_all = np.where( (meanslope<=level[0]))[0]
# verymild_slope = np.random.choice(verymild_slope_all, size=min(len(verymild_slope_all), sample_size), replace=False)
# basinsi_list = [verylarge_slope,large_slope,mild_slope,verymild_slope]
# #climate_label = ['verylarge_slope','large_slope','mild_slope','verymild_slope']
# #climate_label = ['veryhigh_elevation','high_elevation','low_elevation','verylow_elevation']
# climate_label = ['verylarge_area','large_area','small_area','verysmall_area']

# vegetationcover = attributeALL_df["NDVI"].values
# level = [0.3,0.45,0.6]
# verylarge_vegetationcover_all = np.where(vegetationcover>level[2])[0]
# verylarge_vegetationcover = np.random.choice(verylarge_vegetationcover_all, size=min(len(verylarge_vegetationcover_all), sample_size), replace=False)

# large_vegetationcover_all =np.where((vegetationcover<=level[2]) & (vegetationcover>level[1]))[0]
# large_vegetationcover = np.random.choice(large_vegetationcover_all, size=min(len(large_vegetationcover_all), sample_size), replace=False)
# mild_vegetationcover_all =np.where((vegetationcover<=level[1]) & (vegetationcover>level[0]))[0]
# mild_vegetationcover = np.random.choice(mild_vegetationcover_all, size=min(len(mild_vegetationcover_all), sample_size), replace=False)
# verymild_vegetationcover_all = np.where( (vegetationcover<=level[0]))[0]
# verymild_vegetationcover = np.random.choice(verymild_vegetationcover_all, size=min(len(verymild_vegetationcover_all), sample_size), replace=False)
# basinsi_list = [verylarge_vegetationcover,large_vegetationcover,mild_vegetationcover,verymild_vegetationcover]
# climate_label = ['verylarge_vegetationcover','large_vegetationcover','mild_vegetationcover','verymild_vegetationcover']



# Let's assume you have a list of variables that you want to plot,
# and a corresponding list of titles for those plots.
# area_frac = np.repeat(np.expand_dims(np.array(A_change_list), axis = 0),len(attribute),axis = 0) 
# area_exp = np.repeat(np.expand_dims(attributeALL_df["area"].values, axis = -1),len(A_change_list),axis = -1)
# area_frac = area_frac/area_exp
variables = [Bias_area,peak_flow_area, Q2_Q_area, mean_flow_frac_area,]
#titles = ['Q2/Q',r'mean annual flow fraction $\overline{Q}$/$\overline{Q^*}$', r'$Q_{\max}/\overline{Q}$']
titles = [r'Total bias percentage',r'$Q_{\max}/\overline{Q}$', 'Q2/Q', r'$\overline{Q}$/$\overline{Q_{minA}}$']
# Define the number of subplots based on the number of variables
n_subplots = len(variables)
fontsize = 24
plt.rcParams.update({'font.size': fontsize})
# Create a figure and a set of subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 16))  # Adjust figsize as needed
# Define color labels

color_labels = ['red','orange','green','darkblue','cyan','darkblue']
ls_list=["--",":","dotted","dashdot"]
# Loop over each variable and create a subplot
for i, ax in enumerate(axes.flatten()):
    
        
    # Plot the variable on the ith subplot
    # Plot the variable on the ith subplot
    for climatei in range(len(basinsi_list)):
        color = color_labels[climatei]
        # if climatei == 3 or climatei == 4:
        #     for basinsi in basinsi_list[climatei]:
        #         if basinsi in basinsi_list[climatei]:
        #             color = color_labels[climatei]


        #         ax.plot(np.sqrt(np.array(A_change_list)), variables[i][basinsi, :], ls='--', alpha=0.25, c=color)

        
       
        median_line = np.nanpercentile(variables[i][basinsi_list[climatei],:],50,axis = 0)
        upper_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],90,axis = 0)
        lower_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],10,axis = 0)
        ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.2)
        ax.plot(np.sqrt(np.array(A_change_list)),median_line,lw = 2, ls = '-',c =color,label=climate_label[climatei])
    # Set the title of the ith subplot
    #ax.set_title(titles[i])
    
    # Add any other necessary plot customizations here
    #ax.set_xlabel(r'sqrt(Area) (km)')
    if i == 2 or i== 3:
        ax.set_xlabel(r'sqrt(Area) (km)')
    ax.set_ylabel(titles[i])
    if i == 0:
        ax.set_ylim([-0.5,0.5])    
    if i == 3:
        ax.set_ylim([0.5,1.5])
    if i == 1:
        ax.legend(loc='upper right',fontsize = 30)
    ax.set_xlim([0,200])
# Adjust the layout so that titles and labels don't overlap

    
plt.tight_layout()
plt.savefig("Partial_dependence_variables.png", dpi=300)
print("Done")
# Show the plot




# Assuming attributeALL_df is defined
lat = attributeALL_df["lat"].values[very_humid]
lon = attributeALL_df["lon"].values[very_humid]


variables = [Q2_Q_area[very_humid,:], peak_flow_area[very_humid,:], abs(Bias_area[very_humid,:]), mean_flow_frac_area[very_humid,:]]
titles = ['Q2/Q', 'mean annual (Q_max/$\overline{Q}_*$)', 'Absolute total bias percentage', r'$\overline{Q}$/$\overline{Q_{minA}}$']

plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(2, 2, figsize=(20, 20))

for i, ax in enumerate(axes.flatten()):
    m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-65, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x, y = m(lon, lat)
    
    max_element_indices = np.argmax(np.abs(variables[i]-1), axis=1)

    # Extract the first element and the absolute maximum element in each row
    first_elements = variables[i][np.arange(variables[i].shape[0]), 0]  # First elements of each row
    max_elements = variables[i][np.arange(variables[i].shape[0]), max_element_indices]  # Absolute maximum elements of each row


    delta_variable = (max_elements - first_elements) / (np.sqrt(np.array(A_change_list)[max_element_indices])-A_change_list[0]+0.000001)*1000
    Bi_slope =  delta_variable*0
    Bi_slope_10 = np.nanpercentile(delta_variable,20)
    Bi_slope_90 = np.nanpercentile(delta_variable,80)
    Bi_slope[delta_variable<Bi_slope_10 ]= -1
    Bi_slope[delta_variable>Bi_slope_90 ]= 1

    print(len(delta_variable))
    vmin = np.nanpercentile(delta_variable,10)
    vmax = np.nanpercentile(delta_variable,90)
    if i==3:
        scatter = ax.scatter(x, y, s=50, c=Bi_slope, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
        #scatter = ax.scatter(x, y, s=50, c=Bi_slope, cmap=plt.cm.seismic, vmin=-np.maximum(abs(vmin),abs(vmax)), vmax=np.maximum(abs(vmin),abs(vmax)))
    else:
        scatter = ax.scatter(x, y, s=50, c=delta_variable, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    ax.set_title(titles[i])
    plt.colorbar(scatter, ax=ax, fraction = 0.11, location='bottom', pad=0.05, label='(maximum -Amin )/ Amin')

plt.tight_layout()
plt.savefig("Variable_change_map.png", dpi=300)
print("Done")
# Show the plot


correlations = np.array([np.corrcoef(Bi_slope , attribute[very_humid, i])[0, 1] for i in range(attribute.shape[1])])

# Find the column with the maximum absolute correlation with A
max_corr_index = np.nanargmax(np.abs(correlations))

print("Correlations:", correlations)
print("Column with maximum correlation:", attributeLst[max_corr_index])
print("Maximum correlation value:", correlations[max_corr_index])









# Assuming attributeALL_df is defined
lat = attributeALL_df["lat"].values[very_humid]
lon = attributeALL_df["lon"].values[very_humid]



plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(int(len(attributeLst)/2), 2, figsize=(20, int(len(attributeLst)/2)*10))

for i, ax in enumerate(axes.flatten()):
    m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-65, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x, y = m(lon, lat)
    delta_variable = attribute[very_humid,i]
    vmin = np.nanpercentile(delta_variable,2)
    vmax = np.nanpercentile(delta_variable,98)
    scatter = ax.scatter(x, y, s=50, c=delta_variable, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    ax.set_title(attributeLst[i])
    plt.colorbar(scatter, ax=ax, fraction = 0.11, location='bottom', pad=0.05)

plt.tight_layout()
plt.savefig("very_humid_attribute_map.png", dpi=300)
print("Done")









aridity = attributeALL_df["aridity"].values
elevation = attributeALL_df["meanelevation"].values
markers = ['o', 's', 'D', '^', 'v'] 
# Creating the plot
plt.figure(figsize=(10, 8))

sc = plt.scatter(aridity, elevation,  marker=markers[0],c=Bias_area[:,0], cmap='seismic',vmax=0.5,vmin=-0.5)

plt.colorbar(sc, label='Bias')
plt.xlabel('Aridity')
plt.ylabel('Elevation')
plt.title('Scatter Plot of Elevation vs Aridity with Colormap of Bias')
plt.savefig("Elevation vs Aridity.png", dpi=300)
# Display the plot
print("Done")




aridity = attributeALL_df["aridity"].values

nbin = 4
lower_bound = 0
upper_bound = 24000
#bins = np.linspace(lower_bound, upper_bound, nbin + 1)
bin_length = (upper_bound - lower_bound) / (nbin-1)
bins =np.array([0,8000,16000,24000,32000])

bins_split =np.array([0,0.8,1.2,2.5,5])


area_bin_index = np.digitize(aridity, bins_split)


plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
labels = []
for bin_i in range(len(bins_split)-1):
    labels.append(f'{bins_split[bin_i]}~{bins_split[bin_i+1]}')

plot1 = ax.boxplot( [ NSE_base[np.where(area_bin_index == i)][~np.isnan(NSE_base[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+1*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="aliceblue", color="k") ,widths = bin_length/6)
plot2 = ax.boxplot( [ NSE_training_base[np.where(area_bin_index == i)][~np.isnan(NSE_training_base[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+2*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="mediumpurple", color="k"),widths = bin_length/6 )

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
# for whisker in plot3['whiskers']:
#     whisker.set(ls='-', linewidth=2,color = "k")
# for cap in plot3['caps']:
#     cap.set(ls='-', linewidth=2,color = "k")
# for box in plot3['boxes']:
#     box.set(ls='-', linewidth=2)
# for median in plot3['medians']:
#     median.set(ls='-', linewidth=2,color = "k")


y_upper = 1
y_lower = 0
yrange = y_upper-y_lower
for i in range(1,nbin+1):

    num = len(aridity[np.where(area_bin_index == i)])
    ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.1*(y_upper-y_lower), f'{num} sites')

ax.add_patch( Rectangle(( 700, y_lower+0.3*yrange),200, yrange*0.05,  fc = "aliceblue",  ec ='k',ls = "-" , lw = 2) )
ax.text(1000,y_lower+0.3*yrange, r"Forward on validation (1980-1995)")
ax.add_patch( Rectangle(( 700, y_lower+0.2*yrange), 200, yrange*0.05,  fc = "mediumpurple",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.2*yrange, r"Forward on training (1995-2010)")
# ax.add_patch( Rectangle(( 700, y_lower+0.1*yrange), 200, yrange*0.05,  fc = "pink",  ec ='k',ls = "--" , lw = 2) )
# ax.text(1000, y_lower+0.1*yrange, r"Forward on training (1980-1995) selected 2800 Gages")

#ax.set_ylabel("Total bias percentage")
ax.set_ylabel("NSE")
ax.set_xlabel(r"Aridity")

ax.set_yticks(np.arange(y_lower,y_upper,yrange/5))
ax.set_ylim([y_lower,y_upper])
ax.set_xlim([lower_bound,upper_bound+bin_length])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.vlines([-0.5,0,0.5], -2, 4,color ="k",linestyles='--',lw = 1.5)
ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), y_lower,y_upper,color ="k",linestyles='--',lw = 2.5)
tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
ax.set_xticks(tick_positions)
#ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
ax.set_xticklabels(labels)

plt.savefig("boxplot_NSE_validation_aridity.png", dpi=300)
plt.show(block=True)

print("Done")





















titles = np.array(parameters)[np.sort(SubInd)]
# Define the number of subplots based on the number of variables
n_subplots = HBVpara_area.shape[-1]

# Create a figure and a set of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust figsize as needed


color_labels = ['red','orange','green','blue']
ls_list=["--",":","dotted","dashdot"]
# Loop over each variable and create a subplot
for i, ax in enumerate(axes.flatten()):
    
    # Plot the variable on the ith subplot
    # Plot the variable on the ith subplot
    if i<n_subplots:
        for climatei in range(len(basinsi_list)):

            color = color_labels[climatei]
        
            median_line = np.nanpercentile(HBVpara_area[basinsi_list[climatei],:,i],50,axis = 0)
            upper_line  = np.nanpercentile(HBVpara_area[basinsi_list[climatei],:,i],95,axis = 0)
            lower_line  = np.nanpercentile(HBVpara_area[basinsi_list[climatei],:,i],5,axis = 0)
            ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.2)
            ax.plot(np.sqrt(np.array(A_change_list)),median_line,ls = '-',c =color,label=climate_label[climatei])
    # Set the title of the ith subplot


        # Set the title of the ith subplot
        #ax.set_title(titles[i])
        
        # Add any other necessary plot customizations here
        ax.set_xlabel(r'sqrt(Area) (km)')
        ax.set_ylabel(titles[i])


# Adjust the layout so that titles and labels don't overlap
plt.tight_layout()
plt.legend()
plt.savefig("Partial_dependence_parameters.png", dpi=300)
print("Done")





titles = ["rout a","rout b"]
# Define the number of subplots based on the number of variables
n_subplots = Routpara_area.shape[-1]

# Create a figure and a set of subplots
fig, axes = plt.subplots(1, n_subplots, figsize=(10, 5))  # Adjust figsize as needed

# Loop over each variable and create a subplot
for i, ax in enumerate(axes.flatten()):

    if i<n_subplots:
        for climatei in range(len(basinsi_list)):

            color = color_labels[climatei]
        
            median_line = np.nanpercentile(Routpara_area[basinsi_list[climatei],:,i],50,axis = 0)
            upper_line  = np.nanpercentile(Routpara_area[basinsi_list[climatei],:,i],95,axis = 0)
            lower_line  = np.nanpercentile(Routpara_area[basinsi_list[climatei],:,i],5,axis = 0)
            ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.2)
            ax.plot(np.sqrt(np.array(A_change_list)),median_line,ls = '-',c =color,label=climate_label[climatei])


        ax.set_xlabel(r'sqrt(Area) (km)')
        ax.set_ylabel(titles[i])

# Adjust the layout so that titles and labels don't overlap
plt.tight_layout()
plt.savefig("Partial_dependence_route_parameters.png", dpi=300)
print("Done")

# lat =  attributeALL_df["lat"].values
# lon=  attributeALL_df["lon"].values
#



# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import cartopy.crs as ccrs
#
# from mpl_toolkits.basemap import Basemap
#
# NSElst= dataBox[0][0]
# selected_site  =np.where(NSElst>0.4)[0]
# NSElst_selected = NSElst[selected_site]
# deltaHBVpara_selected = deltaHBVpara[selected_site]
# deltaHBVpara_index_selected = deltaHBVpara_index[selected_site]
# deltaHBVparaRout_selected = deltaHBVparaRout[selected_site]
# deltaHBVparaRout_index_selected = deltaHBVparaRout_index[selected_site]
#
# lat_selected  = lat[selected_site]
# lon_selected = lon[selected_site]
# # Load your dataset here, which must include latitude, longitude, correlation, and mean flowrate.
#
#
#
# # Convert latitude and longitude to x and y coordinates
#
# # Plot each point, with size and color determined by mean flowrate and correlation
# parameters = ["Beta","FC","K0","K1","K2","LP","PERC","UZL","TT","CFMAX","CFR","CWH","BETAET"]
# color_names = [
#     'red', 'green', 'blue', 'yellow',
#     'orange', 'purple', 'cyan',
#     'magenta', 'lime', 'pink',
#     'teal', 'olive', 'maroon'
# ]
# # Convert latitude and longitude to x and y coordinates
# # for i, parameter in enumerate(parameters):
# #     # Find indexes for each parameter
# #     indexes = np.where(deltaHBVpara_index == i)[0]
# #     # Convert lat/lon to map projection coordinates
# #     x, y = m(lon[indexes], lat[indexes])
# #     # Plot using a specific color from color_names
# #     for ii,idx in enumerate(indexes):
# #         m.scatter(x[ii], y[ii], s=deltaHBVpara[idx]*200, color=color_names[i % len(color_names)], alpha=0.5, label=parameter)
# fontsize = 14
# plt.rcParams.update({'font.size': fontsize})
#
# figsize = [10, 8]
# fig = plt.figure(figsize=figsize)
# # Create a Basemap instance
# m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50,
#             llcrnrlon=-125, urcrnrlon=-65, resolution='i')
#
# m.drawcoastlines()
# m.drawcountries()
# m.drawstates()
# labels_added = set()
# # selected_parameters = ["Beta","FC","BETAET","LP"]
# selected_parameters = ["K0","K1","K2","PERC","UZL"]
# # selected_parameters =["TT","CFMAX","CFR","CWH"]
# for i, parameter in enumerate(parameters):
#     if parameter in selected_parameters:
#         indexes = np.where(deltaHBVpara_index_selected == i)[0]
#         if len(indexes) > 0:
#             x, y = m(lon_selected[indexes], lat_selected[indexes])
#             sizes = abs(np.max(deltaHBVpara_selected[indexes],axis = -1) )* 20  # Calculate sizes
#
#             # Check and add label only once per parameter
#             label = parameter if parameter not in labels_added else None
#             m.scatter(x, y, s=sizes, color=color_names[i % len(color_names)], alpha=0.5, label=label)
#             labels_added.add(parameter)
#
#
# legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# for legend_handle in legend.legendHandles:
#     legend_handle._sizes = [30]  # Set a standard size for all legend markers
#
# plt.tight_layout()
# plt.show(block=True)
#
# parameters = ["rout a","rout b"]
#
# fontsize = 14
# plt.rcParams.update({'font.size': fontsize})
#
# figsize = [10, 8]
# fig = plt.figure(figsize=figsize)
# # Create a Basemap instance
# m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50,
#             llcrnrlon=-125, urcrnrlon=-65, resolution='i')
#
# m.drawcoastlines()
# m.drawcountries()
# m.drawstates()
# labels_added = set()
# # selected_parameters = ["Beta","FC","BETAET","LP"]
# selected_parameters = ["rout a","rout b"]
# # selected_parameters =["TT","CFMAX","CFR","CWH"]
# for i, parameter in enumerate(parameters):
#     if parameter in selected_parameters:
#         indexes = np.where(deltaHBVparaRout_index_selected == i)[0]
#         if len(indexes) > 0:
#             x, y = m(lon_selected[indexes], lat_selected[indexes])
#             sizes = np.sqrt(abs(np.max(deltaHBVparaRout_selected[indexes],axis = -1) )) *2 # Calculate sizes
#
#             # Check and add label only once per parameter
#             label = parameter if parameter not in labels_added else None
#             m.scatter(x, y, s=sizes, color=color_names[i % len(color_names)], alpha=0.5, label=label)
#             labels_added.add(parameter)
#
#
# legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# for legend_handle in legend.legendHandles:
#     legend_handle._sizes = [30]  # Set a standard size for all legend markers
#
# plt.tight_layout()
# plt.show(block=True)



# fontsize = 18
# plt.rcParams.update({'font.size': fontsize})
# plotTime = pd.date_range('1992-10-01',f'1993-10-01', freq='d', closed='left')
# plt.figure(figsize=(10, 6))
# plt.plot(plotTime,
#          dataPred[np.where(idLst_new==6274300)[0][0],idx_start:idx_end+1,0]*0.0283168,
#          label=f'Forward flow: NSE {round(NSElst[np.where(idLst_new==6274300)[0][0]], 2)}', lw=2, color='red')
# plt.plot(plotTime,
#          streamflow[np.where(idLst_new==6274300)[0][0],idx_start:idx_end+1,0]*0.0283168, label='Observation', lw=2,
#          color='k')
# plt.title(f'Gage 06274300')
# plt.xlabel('Date')
# plt.ylabel(r'Discharge (m$^3$/s)')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show(block=True)