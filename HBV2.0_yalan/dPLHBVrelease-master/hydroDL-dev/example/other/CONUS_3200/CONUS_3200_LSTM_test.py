import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))
from hydroDL.model import rnn, crit, train
from hydroDL.data import camels
from hydroDL.post import plot, stat
from hydroDL.master import loadModel
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
import zarr
import pickle
## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    dfdate = pd.date_range(start=str(trange[0]), end=str(trange[1]), freq='D') # end not included
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
traingpuid = 1
torch.cuda.set_device(traingpuid)


data_folder = "/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/gages/dataCONUS3200/"
# data_folder_selected = "/data/yxs275/CONUS_data/HUC10/version_2_11_25/"
# selected_gageIdx = np.load(data_folder_selected+"selected_gageIdx.npy")


with open(data_folder+'test_data_dict.json') as f:
    train_data_dict = json.load(f)

# forcingAll_test = np.load(data_folder+"test_forcing.npy")
streamflow_test = np.load(data_folder+"test_flow.npy")
# forcingAll_train = np.load(data_folder+"train_forcing.npy")
streamflow_train = np.load(data_folder+"train_flow.npy")
warmup_span = pd.date_range('1980-10-01',f'1995-09-30', freq='d')
Tex = [19801001, 20100930]
warmup_for_testing = len(warmup_span)

AllTime = pd.date_range('1980-01-01', f'2020-12-31', freq='d')


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


gage_info_file_selected_from_merit = "/projects/mhpi/data/MERIT/gage_information/formatted_gage_csvs/gages_3000_merit_info.csv"
gage_info_from_merit = pd.read_csv(gage_info_file_selected_from_merit)


gage_info_from_merit = gage_info_from_merit.sort_values(by='STAID')
gageIDs_from_merit = gage_info_from_merit['STAID'].values

attributeALL_df  = attributeALL_df[attributeALL_df['id'].isin(gageIDs_from_merit)]



attributeAllLst = attributeALL_df.columns

basin_area = np.expand_dims(attributeALL_df["area"].values,axis = 1)
lat =  attributeALL_df["lat"].values
idLst_new = attributeALL_df["id"].values
idLst_old = [int(id) for id in shapeID_str_lst]
[C, ind1, SubInd] = np.intersect1d(idLst_new, idLst_old, return_indices=True)
forcingAll = forcingAll[SubInd,:,:]
streamflow = streamflow[SubInd,:,:]


if(not (idLst_new==np.array(idLst_old)[SubInd]).all()):
   raise Exception("Ids of subset gage do not match with id in the attribtue file")

log_norm_cols=[]

attributeLst = ['area','ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']

attributeLst_old = attributeLst.copy()

stat_dict={}
for fid, forcing_item in enumerate(forcingAllLst) :
    if forcing_item in log_norm_cols:
        stat_dict[forcing_item] = cal_stat_gamma(forcingAll[:,:,fid])
    else:
        stat_dict[forcing_item] = cal_stat(forcingAll[:,:,fid])

[C, _, SubInd] = np.intersect1d(attributeLst, attributeAllLst, return_indices=True)

attribute = attributeALL_df.iloc[ind1, np.sort(SubInd)].values

attributeLst  = list(attributeAllLst[np.sort(SubInd)])

if(not (np.array(attributeLst)==np.array(attributeLst_old)).all()):
   raise Exception("AttributeLst is not in the order provivded")


for aid, attribute_item in enumerate (attributeLst):
    stat_dict[attribute_item] = cal_stat(attribute[:,aid])





tmin = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmin")[0][0]], 0,1)
tmax = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmax")[0][0]], 0,1)

tmean = (tmin+tmax)/2

latarray = np.tile(lat, [tmin.shape[0], 1])
pet = hargreaves(tmin, tmax, tmean, lat=latarray, trange=Tex)


xTrain = np.zeros([forcingAll.shape[0],forcingAll.shape[1],3])
xTrain[:,:,0] = forcingAll[:,:,np.where(np.array(forcingAllLst) == "prcp")[0][0]]
xTrain[:,:,1] = np.swapaxes(tmean, 0,1)
xTrain[:,:,2] = np.swapaxes(pet, 0,1)


#streamflow_trans[streamflow_trans!=streamflow_trans] = 0

stat_dict['tmean'] = cal_stat(xTrain[:,:,1])
stat_dict['pet'] = cal_stat(xTrain[:,:,2])




mean_prep  = attributeALL_df["meanP"].values
mean_prep = np.expand_dims(mean_prep/365,axis =-1) 



# basinAreaName = "DRAIN_SQKM"
# basin_area = attributeGAGEII[:,np.where(np.array(attributeGAGEIILst)=="DRAIN_SQKM")[0]]


rootOut = "/projects/mhpi/yxs275/model/"+'/LSTM_local_daymet_filled_withNaN_NSE_with_same_forcing_HBV_2800/'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, "exp_EPOCH300_BS100_RHO365_HS512_trainBuff365/")  # output folder to save results
#out = os.path.join(rootOut, "exp001_new")
if os.path.exists(out) is False:
    os.mkdir(out)


#with open(rootOut+'/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365dapengscaler_stat.json') as f:
with open(out + '/dapengscaler_stat.json') as f:
    stat_dict = json.load(f)

forcing_LSTM_norm = _trans_norm(
    xTrain,['prcp','tmean','pet']  , stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)
forcing_LSTM_norm[forcing_LSTM_norm!=forcing_LSTM_norm]  = 0

attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
attribute_norm[attribute_norm!=attribute_norm] = 0

testepoch = 300
model_path = out

print("Load model from ", model_path)

testmodel = loadModel(model_path, epoch=testepoch)

testbatch = 300 #len(checkid)#50

filePathLst = [out+"/alltime_out0"]


# train.testModel(
#     testmodel, forcing_LSTM_norm,None, c=attribute_norm, batchSize=testbatch, filePathLst=filePathLst)

dataPred = pd.read_csv(  out+"/alltime_out0", dtype=np.float32, header=None).values
dataPred = np.expand_dims(dataPred, axis=-1)


yPred = _trans_norm(
    dataPred[:, :, 0 :  1].copy(),
    ['usgsFlow'],
    stat_dict,
    log_norm_cols=log_norm_cols,
    to_norm=False,
)

yPred = _basin_norm_for_LSTM(
                        yPred.copy(), basin_area, mean_prep, to_norm=False
                    )


streamflow_mmday = _basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, mean_prep, to_norm=True
                    )

yPred_mmday = _basin_norm(
                        yPred.copy(), basin_area, mean_prep, to_norm=True
                    )

with open('/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/CONUS2800_test.pkl', 'wb') as f:
    pickle.dump((xTrain, streamflow_mmday, attribute), f)

evaDict = [stat.statError(yPred_mmday[:,warmup_for_testing:,0], streamflow_mmday[:,warmup_for_testing:,0])]

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



import xarray as xr
import zarr
key_info = [str(x) for x in idLst_new]
variables_name = ['Qr']
test_time_range = pd.date_range('1980-10-01',f'2010-09-30', freq='d')
data_arrays = {}
for idx, var_x in enumerate(variables_name):

    data_array = xr.DataArray(
        yPred_mmday[:,:,0],
        dims = ['COMID','time'],
        coords = {'COMID':key_info,
                    'time':test_time_range}
    )

    data_arrays[var_x] = data_array
    
    
data_array = xr.DataArray(
    streamflow_mmday[:,:,0],
    dims = ['COMID','time'],
    coords = {'COMID':key_info,
                'time':test_time_range}
)

data_arrays['runoff'] = data_array

data_array = xr.DataArray(
    streamflow[:,:,0],
    dims = ['COMID','time'],
    coords = {'COMID':key_info,
                'time':test_time_range}
)

data_arrays['streamflow'] = data_array
    

xr_dataset = xr.Dataset(data_arrays)
xr_dataset.to_zarr(store=out, group=f'simulation', mode='w')



gage_info_file_selected_from_merit = "/projects/mhpi/data/MERIT/gage_information/formatted_gage_csvs/gages_3000_merit_info.csv"
gage_info_from_merit = pd.read_csv(gage_info_file_selected_from_merit)
gage_info_from_merit = gage_info_from_merit.sort_values(by='STAID')
gageIDs_from_merit = gage_info_from_merit['STAID'].values

attributeALL_df_trained  = attributeALL_df[attributeALL_df['id'].isin(gageIDs_from_merit)]
trained_gages = attributeALL_df_trained['id'].values



[C, ind_merit, SubInd_merit] = np.intersect1d(trained_gages, idLst_new, return_indices=True)

lat = attributeALL_df["lat"].values[SubInd_merit]
lon = attributeALL_df["lon"].values[SubInd_merit]
selected_area = basin_area[SubInd_merit]


if not (trained_gages == np.array(idLst_new)[SubInd_merit]).all():
    raise Exception("IDs of subset gage do not match with ID in the attribute file")


if not (attributeALL_df_trained['area'].values == selected_area[:,0]).all():
    raise Exception("area of subset gage do not match with area in the attribute file")


evaDict = [stat.statError(yPred_mmday[SubInd_merit,warmup_for_testing:,0], streamflow_mmday[SubInd_merit,warmup_for_testing:,0])]
evaDictLst = evaDict
# keyLst = ['NSE', 'KGE']
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

# print("NSE,KGE, dMax: ", np.nanmedian(dataBox[0][0]),np.nanmedian(dataBox[1][0]))
print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))

NSE_LSTM = dataBox[0][0]
FLV_LSTM = dataBox[2][0]
FHV_LSTM = dataBox[3][0]
absFLV_LSTM = dataBox[7][0]
absFHV_LSTM = dataBox[8][0]
out_dHBV_water_loss = "/projects/mhpi/yxs275/model/water_loss_model" + '/dPL_local_daymet_new_attr_water_loss_v3_correct_Ai/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/'


dataPred_water_loss = pd.read_csv(  out_dHBV_water_loss+"/out0", dtype=np.float32, header=None).values
dataPred_water_loss = np.expand_dims(dataPred_water_loss, axis=-1)




evaDict = [stat.statError(dataPred_water_loss[ind_merit,warmup_for_testing:,0], streamflow_mmday[SubInd_merit,warmup_for_testing:,0])]
evaDictLst = evaDict
# keyLst = ['NSE', 'KGE']
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

# print("NSE,KGE, dMax: ", np.nanmedian(dataBox[0][0]),np.nanmedian(dataBox[1][0]))
print("dHBV water loss model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))

NSE_water_loss = dataBox[0][0]
FLV_water_loss = dataBox[2][0]
FHV_water_loss = dataBox[3][0]
absFLV_water_loss = dataBox[7][0]
absFHV_water_loss = dataBox[8][0]

obs = streamflow_mmday[SubInd_merit,warmup_for_testing:,0]

out_dHBV_NSE = "/projects/mhpi/yxs275/model/" + '/dPL_local_daymet_new_attr_NSEloss/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/'


dataPred_dHBV_NSE = pd.read_csv(  out_dHBV_NSE+"/out0", dtype=np.float32, header=None).values
dataPred_dHBV_NSE = np.expand_dims(dataPred_dHBV_NSE, axis=-1)

evaDict = [stat.statError(dataPred_dHBV_NSE[SubInd_merit,-obs.shape[1]:,0], streamflow_mmday[SubInd_merit,warmup_for_testing:,0])]
evaDictLst = evaDict
# keyLst = ['NSE', 'KGE']
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

# print("NSE,KGE, dMax: ", np.nanmedian(dataBox[0][0]),np.nanmedian(dataBox[1][0]))
print("dHBV NSE model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))

NSE_NSE = dataBox[0][0]
FLV_NSE = dataBox[2][0]
FHV_NSE = dataBox[3][0]
absFLV_NSE = dataBox[7][0]
absFHV_NSE = dataBox[8][0]

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


LSTM_pred = yPred_mmday[SubInd_merit,warmup_for_testing:,0]
dHBV_pred_water_loss = dataPred_water_loss[ind_merit,warmup_for_testing:,0]
dHBV_pred_NSE = dataPred_dHBV_NSE[SubInd_merit,warmup_for_testing:,0]







results_savepath = "/projects/mhpi/yxs275/DM_output/" + '/dPL_local_daymet_new_attr_water_loss_v3_correct_Ai_epoch50/CONUS_gage_10%'

simulationroot = zarr.open_group(results_savepath, mode='r')
gages_equal = simulationroot['gage'][:]
gages_equal = [int(x) for x in gages_equal]
merit_pred = simulationroot['simulation_data']
[C, ind_merit_forward_equal, SubInd_merit_forward_equal] = np.intersect1d(gages_equal, trained_gages[ind_merit], return_indices=True)
dHBV_pred_MERIT_equal = merit_pred[ind_merit_forward_equal,-obs.shape[1]:]

if not (np.array(gages_equal)[ind_merit_forward_equal] == np.array(trained_gages[ind_merit])[SubInd_merit_forward_equal]).all():
    raise Exception("IDs of subset gage do not match with ID in the attribute file")



attributeALL_df_validate  = attributeALL_df[attributeALL_df['id'].isin(gages_equal)]
validate_gages = attributeALL_df_validate['id'].values

validate_aridity  = attributeALL_df_validate['aridity'].values

lat_equal = lat[SubInd_merit_forward_equal]
lon_equal= lon[SubInd_merit_forward_equal]
selected_area_equal = selected_area[SubInd_merit_forward_equal]

if not (attributeALL_df_validate['area'].values == selected_area_equal[:,0]).all():
    raise Exception("area of subset gage do not match with area in the attribute file")



LSTM_pred_equal = LSTM_pred[SubInd_merit_forward_equal,:]
dHBV_pred_water_loss_equal = dHBV_pred_water_loss[SubInd_merit_forward_equal,:]
dHBV_pred_NSE_equal = dHBV_pred_NSE[SubInd_merit_forward_equal,:]
obs_mert_equal = obs[SubInd_merit_forward_equal,:]


dHBV_pred_MERIT_equal_aridity = dHBV_pred_MERIT_equal[np.where(validate_aridity>1.5)[0],:]
obs_mert_equal_aridity = obs_mert_equal[np.where(validate_aridity>1.5)[0],:]

evaDict = [stat.statError(dHBV_pred_MERIT_equal, obs_mert_equal)]
evaDictLst = evaDict
# keyLst = ['NSE', 'KGE']
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

# print("NSE,KGE, dMax: ", np.nanmedian(dataBox[0][0]),np.nanmedian(dataBox[1][0]))
print("dHBV merit forward model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))









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
x, y = m(lon_equal, lat_equal)
# Plot each point, with size and color determined by mean flowrate and correlation
time_allyear = pd.date_range(f'{1995}-10-01', f'{2010}-09-30', freq='d')
Bias_LSTM = 0
Bias_dHBV_water_loss = 0
Bias_dHBV_NSE = 0
Bias_dHBV_MERIT = 0
Bias_dHBV_MERIT_aridity = 0
mean_LSTM = 0
mean_dHBV = 0 
mean_dHBV_NSE = 0
mean_dHBV_MERIT =0
mean_dHBV_MERIT_aridity = 0
for year in range(1995,2010):
    time_year = pd.date_range(f'{year}-10-01', f'{year+1}-09-30', freq='d')
    idx_start = time_allyear.get_loc(time_year[0])
    idx_end = time_allyear.get_loc(time_year[-1])

    year_Bias_LSTM,year_mean_LSTM = bias_meanflowratio_calc(LSTM_pred_equal[:,idx_start:idx_end+1],obs_mert_equal[:,idx_start:idx_end+1])
    year_Bias_dHBV_water_loss,year_mean_dHBV = bias_meanflowratio_calc(dHBV_pred_water_loss_equal[:,idx_start:idx_end+1],obs_mert_equal[:,idx_start:idx_end+1])
    year_Bias_dHBV_NSE,year_mean_dHBV_NSE = bias_meanflowratio_calc(dHBV_pred_NSE_equal[:,idx_start:idx_end+1],obs_mert_equal[:,idx_start:idx_end+1])
    year_Bias_dHBV_MERIT,year_mean_dHBV_MERIT = bias_meanflowratio_calc(dHBV_pred_MERIT_equal[:,idx_start:idx_end+1],obs_mert_equal[:,idx_start:idx_end+1])
    year_Bias_dHBV_MERIT_aridity,year_mean_dHBV_MERIT_aridity = bias_meanflowratio_calc(dHBV_pred_MERIT_equal_aridity[:,idx_start:idx_end+1],obs_mert_equal_aridity[:,idx_start:idx_end+1])

    Bias_LSTM = Bias_LSTM + year_Bias_LSTM
    Bias_dHBV_water_loss = Bias_dHBV_water_loss+year_Bias_dHBV_water_loss
    Bias_dHBV_NSE = Bias_dHBV_NSE+year_Bias_dHBV_NSE
    Bias_dHBV_MERIT = Bias_dHBV_MERIT+year_Bias_dHBV_MERIT
    Bias_dHBV_MERIT_aridity = Bias_dHBV_MERIT_aridity+year_Bias_dHBV_MERIT_aridity

    mean_LSTM = mean_LSTM+year_mean_LSTM
    mean_dHBV = mean_dHBV +year_mean_dHBV
    mean_dHBV_NSE = mean_dHBV_NSE+year_mean_dHBV_NSE
    mean_dHBV_MERIT =mean_dHBV_MERIT+year_mean_dHBV_MERIT
    mean_dHBV_MERIT_aridity =mean_dHBV_MERIT_aridity+year_mean_dHBV_MERIT_aridity
Bias_LSTM = Bias_LSTM/15
Bias_dHBV_water_loss = Bias_dHBV_water_loss/15
Bias_dHBV_NSE = Bias_dHBV_NSE/15
Bias_dHBV_MERIT = Bias_dHBV_MERIT/15
Bias_dHBV_MERIT_aridity = Bias_dHBV_MERIT_aridity/15

mean_LSTM = mean_LSTM/15
mean_dHBV = mean_dHBV/15
mean_dHBV_NSE = mean_dHBV_NSE/15
mean_dHBV_MERIT =mean_dHBV_MERIT/15
mean_dHBV_MERIT_aridity =mean_dHBV_MERIT_aridity/15

delta_Bias_equal = abs(Bias_dHBV_water_loss) - abs(Bias_dHBV_MERIT)
#delta_Bias = mean_dHBV_MERIT-delta_Bias_dHBV_water_loss
#delta_Bias = mean_LSTM-mean_dHBV
#delta_Bias=delta_Bias[largeBasins]
scatter = m.scatter(x, y, s=selected_area_equal[:,0] /30, c=delta_Bias_equal, cmap='jet',vmin=-5, vmax=5,alpha = 0.5)

# Create an axes for the colorbar
#cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])

# Create a colorbar in the specified axes
#label = r'mean_LSTM_ratio-mean_dHBV_ratio (ratio = mean simulation / mean obs)'
#label = r'abs(LSTM bias) - abs(HBV bias)'
#label = r'dHBV absolute bias with regional flow - dHBV bias absolute with NSE loss'
label = "abs(direct forward bias with regional flow)-abs(merit forward bias with regional flow)"
plt.colorbar(scatter, pad=0.05,fraction = 0.11, location='bottom', label=label)

plt.tight_layout()
plt.savefig("Map_area_LSTM_bias_observation.png", dpi=300)


print("Done")


from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
nbin = 4
lower_bound = 0
upper_bound = 24000
#bins = np.linspace(lower_bound, upper_bound, nbin + 1)
bin_length = (upper_bound - lower_bound) / (nbin-1)
bins =np.array([0,8000,16000,24000,32000])

#bins_split =np.array([0,500,1000,3000,100000])
bins_split =np.array([0,0.75,1,2,10])
#bins_split = bins
validate_aridity
area_bin_index = np.digitize(validate_aridity, bins_split)
#area_bin_index = np.digitize(selected_area_equal[:,0], bins_split)
#area_bin_index_aridity = np.digitize(selected_area_equal[np.where(validate_aridity>1.5)[0],0], bins_split)

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
labels = []
for bin_i in range(len(bins_split)-1):
    labels.append(f'{bins_split[bin_i]}~{bins_split[bin_i+1]}')

plot1 = ax.boxplot( [ Bias_LSTM[np.where(area_bin_index == i)][~np.isnan(Bias_LSTM[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+1*bin_length/6.0,patch_artist=True,boxprops=dict(facecolor="aliceblue", color="k") ,widths = bin_length/7)
plot2 = ax.boxplot( [ Bias_dHBV_water_loss[np.where(area_bin_index == i)][~np.isnan(Bias_dHBV_water_loss[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+2*bin_length/6.0,patch_artist=True,boxprops=dict(facecolor="mediumpurple", color="k"),widths = bin_length/7 )
plot3 = ax.boxplot( [ Bias_dHBV_MERIT[np.where(area_bin_index == i)][~np.isnan(Bias_dHBV_MERIT[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+3*bin_length/6.0,patch_artist=True,boxprops=dict(facecolor="pink", color="k"),widths = bin_length/7 )
plot4 = ax.boxplot( [ Bias_dHBV_NSE[np.where(area_bin_index == i)][~np.isnan(Bias_dHBV_NSE[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+4*bin_length/6.0,patch_artist=True,boxprops=dict(facecolor="red", color="k"),widths = bin_length/7 )
#plot4 = ax.boxplot( [ Bias_dHBV_MERIT_aridity[np.where(area_bin_index_aridity == i)][~np.isnan(Bias_dHBV_MERIT_aridity[np.where(area_bin_index_aridity == i)])] for i in range(1,nbin+1) ], whis =[5, 95], vert=True,showfliers=True, positions=bins[:-1]+5*bin_length/6.0,patch_artist=True,boxprops=dict(facecolor="green", color="k"),widths = bin_length/7 )

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

y_upper = 3
y_lower = -3
yrange = y_upper-y_lower
for i in range(1,nbin+1):

    num = len(np.where(area_bin_index == i)[0])
    ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.1*(y_upper-y_lower), f'{num} sites')

    # num_aridity = len(np.where(area_bin_index_aridity == i)[0])
    # ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.2*(y_upper-y_lower), f'{num_aridity} arid sites ')


ax.add_patch( Rectangle(( 700, y_lower+0.4*yrange),200, yrange*0.05,  fc = "aliceblue",  ec ='k',ls = "-" , lw = 2) )
ax.text(1000,y_lower+0.4*yrange, r"LSTM")
ax.add_patch( Rectangle(( 700, y_lower+0.3*yrange), 200, yrange*0.05,  fc = "mediumpurple",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.3*yrange, r"dHBV_direct_forward_with_regional_flow")
ax.add_patch( Rectangle(( 700, y_lower+0.2*yrange), 200, yrange*0.05,  fc = "pink",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.2*yrange, r"dHBV_merit_forward_with_regional_flow")
ax.add_patch( Rectangle(( 700, y_lower+0.1*yrange), 200, yrange*0.05,  fc = "red",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.1*yrange, r"dHBV_direct_forward_w/o_regional_flow")
# ax.add_patch( Rectangle(( 700, y_lower+0*yrange), 200, yrange*0.05,  fc = "green",  ec ='k',ls = "--" , lw = 2) )
# ax.text(1000, y_lower+0.0*yrange, r"dHBV_merit_forward_with_regional_flow for basins (aridity > 1.5)")

ax.set_ylabel("Annal mean bias percentage")
#ax.set_ylabel("NSE")
ax.set_xlabel(r"Aridity")

ax.set_yticks(np.arange(y_lower,y_upper,yrange/10))
ax.set_ylim([y_lower,y_upper])
ax.set_xlim([lower_bound,upper_bound+bin_length])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.hlines(0, 0, 1000000,color ="k",linestyles='--',lw = 1.5)
ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), y_lower,y_upper,color ="k",linestyles='--',lw = 2.5)
tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
ax.set_xticks(tick_positions)
#ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
ax.set_xticklabels(labels)
plt.tight_layout()
plt.savefig("boxplot_bias_area.png", dpi=300)
plt.show(block=True)

print("Done")