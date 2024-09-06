import sys
sys.path.append('../../')

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
    dfdate = pd.date_range(start=str(trange[0]), end=str(trange[1]), freq='D', closed='left') # end not included
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


## GPU setting
# which GPU to use when having multiple
traingpuid = 7
torch.cuda.set_device(traingpuid)


data_folder = "/data/yxs275/DPL_HBV/CONUS_3700_data/generate_for_deltaHBV/gages/dataCONUS3700/"

time = pd.date_range('1980-10-01',f'1995-10-01', freq='d', closed='left')
with open(data_folder+'train_data_dict.json') as f:
    train_data_dict = json.load(f)

forcingAll = np.load(data_folder+"train_forcing.npy")
attributeALL  = np.load(data_folder+"train_attr.npy")
streamflow = np.load(data_folder+"train_flow.npy")
forcingAllLst  = train_data_dict['relevant_cols']
attributeAllLst  = train_data_dict['constant_cols']
lat = attributeALL[:,np.where(np.array(attributeAllLst)=="LAT_GAGE")[0][0]]

log_norm_cols=[]
# log_norm_cols=['prcp', 'pr', 'total_precipitation', 'pre', 'potential_evaporation', 'LE',
#                    'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'ET_water', 'ET_sum', 'susm', 'smp', 'ssma', 'susma',
#                    'usgsFlow', 'streamflow', 'qobs']
forcing_LSTM_List = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
attributeLst = [ "DRAIN_SQKM", "ELEV_MEAN_M_BASIN",
               "SLOPE_PCT", "DEVNLCD06", "FORESTNLCD06", "PLANTNLCD06", "WATERNLCD06", "SNOWICENLCD06","BARRENNLCD06","SHRUBNLCD06",
               "GRASSNLCD06","WOODYWETNLCD06","EMERGWETNLCD06","AWCAVE", "PERMAVE", "RFACT",
               "ROCKDEPAVE","GEOL_REEDBUSH_DOM","GEOL_REEDBUSH_DOM_PCT", "STREAMS_KM_SQ_KM", "NDAMS_2009", "STOR_NOR_2009","RAW_DIS_NEAREST_MAJ_DAM",
               "CANALS_PCT","RAW_DIS_NEAREST_CANAL", "FRESHW_WITHDRAWAL", "POWER_SUM_MW", "PDEN_2000_BLOCK", "ROADS_KM_SQ_KM", "IMPNLCD06",]

#forcing_HBV_List = ['prcp',  'tmax', 'tmin', ]

stat_dict={}
for fid, forcing_item in enumerate(forcingAllLst) :
    if forcing_item in log_norm_cols:
        stat_dict[forcing_item] = cal_4_stat_inds(np.log10(forcingAll[:,:,fid] + 0.1))
    else:
        stat_dict[forcing_item] = cal_4_stat_inds(forcingAll[:,:,fid])

for aid, attribute_item in enumerate (attributeAllLst):
    stat_dict[attribute_item] = cal_4_stat_inds(attributeALL[:,aid])


[C, ind1, SubInd] = np.intersect1d(forcing_LSTM_List, forcingAllLst, return_indices=True)
forcing_LSTM = forcingAll[:, :,SubInd]
forcing_LSTM[forcing_LSTM!=forcing_LSTM] = 0

[C, ind1, SubInd] = np.intersect1d(attributeLst, attributeAllLst, return_indices=True)
attribute = attributeALL[:, SubInd]
attribute[attribute!=attribute] = 0



mean_prep  = attributeALL[:,np.where(np.array(attributeAllLst)=='PPTAVG_BASIN')[0]]
mean_prep = mean_prep/365*10

basinAreaName = "DRAIN_SQKM"
basin_area = attributeALL[:,np.where(np.array(attributeAllLst)=="DRAIN_SQKM")[0]]


streamflow_trans = _basin_norm_for_LSTM(
                        streamflow[:, :, 0 :  1].copy(), basin_area, mean_prep, to_norm=True
                    )  ## from ft^3/s to non-unit

streamflow_trans[streamflow_trans!=streamflow_trans] = 0

stat_dict['usgsFlow'] = cal_4_stat_inds(streamflow_trans)


streamflow_norm = _trans_norm(
    streamflow_trans,
    ['usgsFlow'],
    stat_dict,
    log_norm_cols=log_norm_cols,
    to_norm=True,
)


forcing_LSTM_norm = _trans_norm(
    forcing_LSTM,forcing_LSTM_List , stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)

attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)



EPOCH = 300 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 20
HIDDENSIZE = 256
trainBuff = 0
nx = forcing_LSTM_norm.shape[-1] + attribute_norm.shape[-1]  # update nx, nx = nx + nc
ny = streamflow_norm.shape[-1]

# load model for training
if torch.cuda.is_available():
    model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
else:
    model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)

rootOut = "/data/yxs275/DPL_HBV/CONUS_3700_Output/"+'/LSTM/'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, "exp001") # output folder to save results
if os.path.exists(out) is False:
    os.mkdir(out)

with open(rootOut+'dapengscaler_stat.json','w') as f:
    json.dump(stat_dict, f)

lossFun = crit.RmseLoss()

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
    bufftime=trainBuff)