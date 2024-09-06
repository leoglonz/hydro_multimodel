import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))


from hydroDL.model import rnn, crit, train
from hydroDL.data import camels
from hydroDL.post import plot, stat
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

time = pd.date_range('1980-10-01',f'1995-09-30', freq='d')
with open(data_folder+'train_data_dict.json') as f:
    train_data_dict = json.load(f)

# with open('/data/tkb5476/hydroDL/stat_files/Statistics_basinnorm_3000-gages-sim-3000-5.json') as f_tadd:
#     stat_dict_tadd = json.load(f_tadd)

#BasinId = train_data_dict["sites_id"]
# np.savetxt(data_folder+"BasinId.txt", BasinId,  fmt='%s', delimiter=' ')
AllTime = pd.date_range('1980-01-01', f'2020-12-31', freq='d')
index_start = AllTime.get_loc(time[0])
index_end = AllTime.get_loc(time[-1])
#forcingAll = np.load(data_folder+"train_forcing.npy")
forcingAll = np.load("/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/allBasin_localDaymet.npy")[:,index_start:index_end+1,:]
shapeID_str_lst= np.load("/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/shapeID_str_lst.npy")

forcingAll = fill_Nan(forcingAll)


attributeGAGEII  = np.load(data_folder+"train_attr.npy")
streamflow = np.load(data_folder+"train_flow.npy")
#forcingAllLst  = train_data_dict['relevant_cols']
forcingAllLst  = ['prcp', 'tmax', 'tmin']
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

basin_area = np.expand_dims(attributeALL_df["area"].values,axis = 1)
lat =  attributeALL_df["lat"].values
idLst_new = attributeALL_df["id"].values
idLst_old = [int(id) for id in shapeID_str_lst]
[C, ind1, SubInd] = np.intersect1d(idLst_new, idLst_old, return_indices=True)
forcingAll = forcingAll[SubInd,:,:]
streamflow = streamflow[SubInd,:,:]

[C, ind1, SubInd] = np.intersect1d(idLst_new, idLst_old, return_indices=True)
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



Tex = [19801001, 19950930]

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


streamflow_trans = _basin_norm_for_LSTM(
                        streamflow[:, :, 0 :  1].copy(), basin_area, mean_prep, to_norm=True
                    )  ## from ft^3/s to non-unit

streamflow_trans_mm_day = _basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


# with open('/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/CONUS2800.pkl', 'wb') as f:
#     pickle.dump((xTrain, streamflow_trans_mm_day, attribute), f)


if 'usgsFlow' in log_norm_cols:
    stat_dict['usgsFlow'] = cal_stat_gamma(streamflow_trans)
else:
    stat_dict['usgsFlow'] = cal_stat(streamflow_trans)


streamflow_norm = _trans_norm(
    streamflow_trans,
    ['usgsFlow'],
    stat_dict,
    log_norm_cols=log_norm_cols,
    to_norm=True,
)


forcing_LSTM_norm = _trans_norm(
    xTrain,['prcp','tmean','pet'] , stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)

attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
forcing_LSTM_norm[forcing_LSTM_norm!=forcing_LSTM_norm] = 0
attribute_norm[attribute_norm!=attribute_norm] = 0


EPOCH = 300 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 20
HIDDENSIZE = 512
trainBuff = 365
nx = forcing_LSTM_norm.shape[-1] + attribute_norm.shape[-1]  # update nx, nx = nx + nc
ny = streamflow_norm.shape[-1]

# load model for training
if torch.cuda.is_available():
    model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
else:
    model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)


rootOut = "/projects/mhpi/yxs275/model/"+'/LSTM_local_daymet_filled_withNaN_NSE_with_same_forcing_HBV_remove_small_basins/'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, f"exp_EPOCH{EPOCH}_BS{BATCH_SIZE}_RHO{RHO}_HS{HIDDENSIZE}_trainBuff{trainBuff}") # output folder to save results
if os.path.exists(out) is False:
    os.mkdir(out)

with open(out+'/dapengscaler_stat.json','w') as f:
    json.dump(stat_dict, f)



lossFun = crit.NSELossBatch(np.nanstd(streamflow_norm, axis=1))

model = train.trainModel(
    model,
    forcing_LSTM_norm,
    streamflow_norm,
    attribute_norm,
    lossFun,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=trainBuff,
   )