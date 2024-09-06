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
traingpuid = 7
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


# gage_info_file_selected_from_merit = "/projects/mhpi/data/MERIT/gage_information/formatted_gage_csvs/gages_3000_merit_info.csv"
# gage_info_from_merit = pd.read_csv(gage_info_file_selected_from_merit)


# gage_info_from_merit = gage_info_from_merit.sort_values(by='STAID')
# gageIDs_from_merit = gage_info_from_merit['STAID'].values

# attributeALL_df  = attributeALL_df[attributeALL_df['id'].isin(gageIDs_from_merit)]

attributeAllLst = attributeALL_df.columns

basin_area = np.expand_dims(attributeALL_df["area"].values,axis = 1)
lat =  attributeALL_df["lat"].values
idLst_new = attributeALL_df["id"].values
idLst_old = [int(id) for id in shapeID_str_lst]
[C, ind1, SubInd_id] = np.intersect1d(idLst_new, idLst_old, return_indices=True)
forcingAll = forcingAll[SubInd_id,:,:]
streamflow = streamflow[SubInd_id,:,:]
if(not (idLst_new==np.array(idLst_old)[SubInd_id]).all()):
   raise Exception("Ids of subset gage do not match with id in the attribtue file")

# log_norm_cols= []
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
attributeLst_old = attributeLst.copy()


[C, _, SubInd] = np.intersect1d(attributeLst, attributeAllLst, return_indices=True)
attribute = attributeALL_df.iloc[ind1, np.sort(SubInd)].values

attributeLst  = list(attributeAllLst[np.sort(SubInd)])

if(not (np.array(attributeLst)==np.array(attributeLst_old)).all()):
   raise Exception("AttributeLst is not in the order provided")

T_increase = 0
tmin = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmin")[0][0]], 0,1) + T_increase
tmax = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmax")[0][0]], 0,1) + T_increase



tmean = (tmin+tmax)/2
attribute[:,np.where(np.array(attributeLst) == "meanTa")[0][0]] = attribute[:,np.where(np.array(attributeLst) == "meanTa")[0][0]]+ T_increase


latarray = np.tile(lat, [tmin.shape[0], 1])
pet = hargreaves(tmin, tmax, tmean, lat=latarray, trange=Tex)


P_increase = 0.01

xTrain = np.zeros([forcingAll.shape[0],forcingAll.shape[1],3])
xTrain[:,:,0] = forcingAll[:,:,np.where(np.array(forcingAllLst) == "prcp")[0][0]] * (1+P_increase)
xTrain[:,:,1] = np.swapaxes(tmean, 0,1)
xTrain[:,:,2] = np.swapaxes(pet, 0,1)
attribute[:,np.where(np.array(attributeLst) == "meanP")[0][0]] = attribute[:,np.where(np.array(attributeLst) == "meanP")[0][0]] *  (1+P_increase)


# xTrain[xTrain!=xTrain]  = 0


streamflow_trans = _basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


rootOut ="/projects/mhpi/yxs275/model/scale_analysis/"  + '/dPL_local_daymet_new_attr_NSE_loss_w_log_with_Cr/'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, "exp_EPOCH100_BS100_RHO365_HS512_trainBuff365")  # output folder to save results
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




testepoch = 100
model_path = out
print("Load model from ", model_path)
testmodel = loadModel(model_path, epoch=testepoch)

zTest = np.concatenate([forcing_LSTM_norm, attribute_norm_exp], 2)  # Add attributes to historical forcings as the inversion part
xTest = xTrain
testTuple = (xTest, zTest)
testbatch =600 #len(indexes)

A_change_list = [5,10,20,30,50,80,100,200,300,500,800,1000,1200,1500,1800,2000,3000,4000,5000,6000,8000,10000,15000,20000,30000,40000]

prediction_out = out+f"/P_increase_0.01"
if os.path.exists(prediction_out) is False:
    os.mkdir(prediction_out)
#filePathLst = [prediction_out+"/allBasinParaHBV_base_new",prediction_out+"/allBasinParaRout_base_new",prediction_out+"/allBasinFluxes_base_new"]

filePathLst = [prediction_out + f"/allBasinParaHBV_area_new_base", prediction_out + f"/allBasinParaRout_area_new_base",prediction_out+f"/allBasinFluxes_area_new_base"]


testmodel.inittime = 0
allBasinFluxes_base,allBasinParaHBV_base,allBasinParaRout_base = train.visualParameters(
    testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst,BufferLenth = warmup_for_testing)

allBasinParaHBV_base = np.load(filePathLst[0]+".npy")
allBasinParaRout_base = np.load(filePathLst[1]+".npy")
allBasinFluxes_base = np.load(filePathLst[2]+".npy")[:,-streamflow_test.shape[1]:,:]

dataPred = allBasinFluxes_base[:,:,0]
test_obs = streamflow_trans[:,warmup_for_testing:,0]
evaDict = [stat.statError(dataPred[:,:], test_obs)]
evaDictLst = evaDict
keyLst = ['NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
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

    print("Working on area", A_change_list[id_change])
    attribute[:, area_idx] = A_change_list[id_change]  #*save_col
    attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
    attribute_norm[attribute_norm != attribute_norm] = 0
    
    attribute_norm_exp = np.expand_dims(attribute_norm, axis=1)
    attribute_norm_exp = np.repeat(attribute_norm_exp, forcing_LSTM_norm.shape[1], axis=1)
    
    zTest = np.concatenate([forcing_LSTM_norm, attribute_norm_exp],    2)  # Add attributes to historical forcings as the inversion part
    xTest = xTrain
    testTuple = (xTest, zTest)
    filePathLst = [prediction_out + f"/allBasinParaHBV_area_new_{A_change_list[id_change]}", prediction_out + f"/allBasinParaRout_area_new_{A_change_list[id_change]}",prediction_out+f"/allBasinFluxes_area_new_{A_change_list[id_change]}"]

    allBasinFluxes,allBasinParaHBV, allBasinParaRout = train.visualParameters(
        testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst, BufferLenth=warmup_for_testing)
    # allBasinParaHBV = np.load(filePathLst[0]+".npy")
    # allBasinParaRout = np.load(filePathLst[1]+".npy")
    # allBasinFluxes = np.load(filePathLst[2] + ".npy")[:,-streamflow_test.shape[1]:,:]




    # Q2_Q = np.sum(allBasinFluxes[:,:,3],axis = 1)/(np.sum(allBasinFluxes[:,:,0],axis = 1)+0.00001)


    # Bias,_ =bias_meanflowratio_calc(allBasinFluxes[:,:,0],test_obs)

    # mean_flow = np.full((len(allBasinFluxes)),0)
    # peak_flow = np.full((len(allBasinFluxes)),0)

    # mean_flow_vol = np.full((len(allBasinFluxes)),0)
    # peak_flow_vol = np.full((len(allBasinFluxes)),0)

    # for year in range(1995,2010):
    #     time_year = pd.date_range(f'{year}-10-01', f'{year+1}-09-30', freq='d')
    #     idx_start = time_allyear.get_loc(time_year[0])
    #     idx_end = time_allyear.get_loc(time_year[-1])

    #     peak_flow = peak_flow+np.max(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1)/(np.sum(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1)+0.00001)
    #     peak_flow_vol = peak_flow_vol + _basin_norm_to_m3_s(np.max(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1,keepdims=True),basin_area=np.array(A_change_list[id_change:id_change+1]),to_norm=False)[:,0]

    # mean_flow_vol = mean_flow_vol + _basin_norm_to_m3_s(np.mean(allBasinFluxes[:,:,0],axis = 1,keepdims=True),basin_area=np.array(A_change_list[id_change:id_change+1]),to_norm=False)[:,0]
    # _, mean_flow_ratio = bias_meanflowratio_calc(allBasinFluxes[:,:,0],allBasinFluxes_base[:,:,0])
    # mean_flow = mean_flow+mean_flow_ratio

    # peak_flow_frac = peak_flow/len(range(1995,2010))
    # Q2_Q_area[:,id_change] = Q2_Q
    # Bias_area[:,id_change] = Bias
    # peak_flow_area[:,id_change] = peak_flow_frac
    # mean_flow_frac_area[:,id_change] = mean_flow

    # peak_flow_vol_area[:,id_change] = peak_flow_vol/len(range(1995,2010))
    # mean_flow_vol_area[:,id_change] = mean_flow_vol

    # [C, Ind, SubInd] = np.intersect1d(selected_parameters, parameters, return_indices=True)
    # HBVpara_area[:,id_change,:] = allBasinParaHBV[:, np.sort(SubInd)]
    # Routpara_area[:,id_change, :] = allBasinParaRout
    # # attribute[:,  area_idx] = save_col

