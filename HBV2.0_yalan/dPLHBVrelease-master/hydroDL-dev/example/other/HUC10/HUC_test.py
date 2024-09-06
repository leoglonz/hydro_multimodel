import sys
sys.path.append('../../')

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

## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


data_folder = "/data/yxs275/CONUS_data/HUC10/version_2_11_25/"

attributeLst = ["mean_slope", "mean_aspect", "mean_elev", "dom_land_cover",
                "dom_land_cover_frac", "forest_fraction", "root_depth_50", "root_depth_99", "soil_depth_m",
                "ksat_0_5", "ksat_5_15", "theta_s_0_5", "theta_s_5_15", "theta_r_0_5", "theta_r_5_15", "ksat_0_15_ave",
                "theta_s_0_15_ave", "theta_r_0_15_ave", "ksat_0_5_e", "ksat_5_15_e", "ksat_0_15_ave_e",
                "Porosity", "Permeability_Permafrost", "MAJ_NDAMS", "general_purpose", "max_normal_storage",
                "std_norm_storage", 'NDAMS_2009', 'STOR_NOR_2009']


attributeALL_df = pd.read_csv(data_folder + "attributes.csv")
# attributeALLLst = attributeALL_df.columns[1:]
# attributeALL = attributeALL_df.values[selected_huc10Idx,1:]
#
# [C, ind1, SubInd] = np.intersect1d(attributeLst, attributeALLLst, return_indices=True)
# attribute_selected = attributeALL[:, np.sort(SubInd)]

basinID = attributeALL_df.gage_ID.values
batchSize = 1000
iS = np.arange(0, len(basinID), batchSize)
iE = np.append(iS[1:], len(basinID))
# for item in range(len(iS)):
#     forcingBatch = np.load(data_folder + f"forcings_{iS[item]}_{iE[item]}.npy")
#     if item ==0:
#         forcingAll = forcingBatch
#     else:
#         forcingAll = np.concatenate((forcingAll,forcingBatch),axis = 0)
# forcingAllLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
# log_norm_cols = ['prcp', 'pr', 'total_precipitation', 'pre', 'LE',
#                  'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'susm', 'smp', 'ssma', 'susma',
#                  'usgsFlow', 'streamflow', 'qobs']
# stat_dict={}
# for fid, forcing_item in enumerate(forcingAllLst) :
#     if forcing_item in log_norm_cols:
#         stat_dict[forcing_item] = cal_stat_gamma(forcingAll[:,:,fid])
#     else:
#         stat_dict[forcing_item] = cal_stat(forcingAll[:,:,fid])
for item in range(len(iS)):

    attributeBatch_file = data_folder+f"attributes_{iS[item]}_{iE[item]}.csv"
    attributeBatch_df = pd.read_csv(attributeBatch_file)
    attributeBatchLst = attributeBatch_df.columns[1:]
    attributeBatch = attributeBatch_df.values[:,1:]


    attributeBatch[np.where(attributeBatch == -999)]=np.nan

    #forcingBatchLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    forcingBatchLst =  ['prcp','tmax', 'tmin']
    forcingBatch  =  np.load(data_folder + f"forcings_{iS[item]}_{iE[item]}.npy")
    forcingBatch = fill_Nan(forcingBatch)
    # forcingBatch = np.load("/data/yxs275/CONUS_data/HUC10/version_2_11_25/HUC10_forcing_selected.npy")
    # forcingBatch_gage = np.load("/data/yxs275/CONUS_data/HUC10/version_2_11_25/GAGEII_forcing_selected.npy")
    #forcingBatch[:,1:,1] = forcingBatch[:,0:-1,1]

    Tex = [19800101, 20210101]
    #time = pd.date_range('1980-01-01',f'2020-12-31', freq='d')
    #start_idx = time.get_loc('1980-10-01')
    lat = attributeBatch[:,np.where(np.array(attributeBatchLst)=="lat")[0][0]]



    log_norm_cols=['prcp', 'pr', 'total_precipitation', 'pre',  'LE',
                       'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'susm', 'smp', 'ssma', 'susma',
                       'usgsFlow', 'streamflow', 'qobs']
    #forcing_LSTM_List = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

    attributeLst = [ "area","mean_slope", "mean_elev","dom_land_cover",
                    "dom_land_cover_frac", "forest_fraction","root_depth_50","root_depth_99","soil_depth_m",
                    "ksat_0_5","ksat_5_15","theta_s_0_5","theta_s_5_15","theta_r_0_5","theta_r_5_15","ksat_0_15_ave",
                 "theta_s_0_15_ave", "theta_r_0_15_ave","ksat_0_5_e","ksat_5_15_e","ksat_0_15_ave_e",
                 "Porosity","Permeability_Permafrost", "MAJ_NDAMS","general_purpose", "max_normal_storage",
                "std_norm_storage",'NDAMS_2009', 'STOR_NOR_2009']

    forcing_HBV_List = ['prcp',  'tmax', 'tmin', ]



    [C, ind1, SubInd] = np.intersect1d(attributeLst, attributeBatchLst, return_indices=True)
    attribute = attributeBatch[:, np.sort(SubInd)]


    tmin = np.swapaxes(forcingBatch[:,:,np.where(np.array(forcingBatchLst) == "tmin")[0][0]], 0,1)
    tmax = np.swapaxes(forcingBatch[:,:,np.where(np.array(forcingBatchLst) == "tmax")[0][0]], 0,1)

    tmean = (tmin+tmax)/2

    latarray = np.tile(lat, [tmin.shape[0], 1])
    pet = hargreaves(tmin, tmax, tmean, lat=latarray, trange=Tex)

    xTrain = np.zeros([forcingBatch.shape[0],forcingBatch.shape[1],3])
    xTrain[:,:,0] = forcingBatch[:,:,np.where(np.array(forcingBatchLst) == "prcp")[0][0]]
    xTrain[:,:,1] = np.swapaxes(tmean, 0,1)
    xTrain[:,:,2] = np.swapaxes(pet, 0,1)


    rootOut = "/data/yxs275/DPL_HBV/CONUS_3200_Output/" + '/dPL/'
    if os.path.exists(rootOut) is False:
        os.mkdir(rootOut)
    out = os.path.join(rootOut, "exp_EPOCH50_BS100_RHO365_HS256_trainBuff365")  # output folder to save results
    if os.path.exists(out) is False:
        os.mkdir(out)

    with open(out + '/dapengscaler_stat.json') as f:
        stat_dict = json.load(f)


    forcing_LSTM_norm = _trans_norm(
        xTrain.copy(), ['prcp','tmean','pet'], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
    )
    forcing_LSTM_norm [forcing_LSTM_norm != forcing_LSTM_norm] = 0
    xTrain[xTrain != xTrain] = 0
    attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
    attribute_norm[attribute_norm != attribute_norm] = 0
    attribute_norm = np.expand_dims(attribute_norm, axis=1)
    attribute_norm = np.repeat(attribute_norm, forcing_LSTM_norm.shape[1], axis=1)



    Ninv = forcing_LSTM_norm.shape[-1]+attribute_norm.shape[-1]
    EPOCH = 50 # total epoches to train the mode
    BATCH_SIZE = 100
    RHO = 365
    saveEPOCH = 10
    alpha = 0.25
    HIDDENSIZE = 256
    BUFFTIME = 365 # for each training sample, to use BUFFTIME days to warm up the states.
    routing = True # Whether to use the routing module for simulated runoff
    Nmul = 16 # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
    comprout = False # True is doing routing for each component
    compwts = False # True is using weighted average for components; False is the simple mean
    pcorr = None # or a list to give the range of precip correc

    tdRep = [1, 13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
    tdRepS = [str(ix) for ix in tdRep]
    # ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
    # Must set below ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
    # If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
    ETMod = True
    Nfea = 13  # should be 13 when setting ETMod=True. 12 when ETMod=False
    dydrop = 0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
    staind = -1  # which time step to use from the learned para time series for those static parameters

    model = rnn.MultiInv_HBVTDModel(ninv=Ninv, nfea=Nfea, nmul=Nmul, hiddeninv=HIDDENSIZE, inittime=BUFFTIME,
                                    routOpt=routing, comprout=comprout, compwts=compwts, staind=staind, tdlst=tdRep,
                                    dydrop=dydrop, ETMod=ETMod)
    lstm = model.lstm


    lossFun = crit.RmseLossComb(alpha=alpha)

    forcTuple = [xTrain,forcing_LSTM_norm]



    testepoch = 50
    model_path = out
    print("Load model from ", model_path)
    testmodel = loadModel(model_path, epoch=testepoch)

    zTest = np.concatenate([forcing_LSTM_norm, attribute_norm], 2)  # Add attributes to historical forcings as the inversion part
    xTest = xTrain
    testTuple = (xTest, zTest)
    testbatch =50 #len(indexes)

    results_savepath = "/data/yxs275/DPL_HBV/HUC10_Output/" + '/dPL/'
    if os.path.exists(results_savepath) is False:
        os.mkdir(results_savepath)

    filePathLst = [results_savepath+f"/Qr_{iS[item]}_{iE[item]}",results_savepath+f"/Q0_{iS[item]}_{iE[item]}",results_savepath+f"/Q1_{iS[item]}_{iE[item]}",results_savepath+f"/Q2_{iS[item]}_{iE[item]}",results_savepath+f"/ET_{iS[item]}_{iE[item]}"]

    testmodel.inittime = 0
    train.testModel(
        testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst)
    # filePathLst = [results_savepath + f"/para_{iS[item]}_{iE[item]}.npy", results_savepath + f"/route_{iS[item]}_{iE[item]}.npy"]
    # train.visualParameters(
    #     testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst,BufferLenth = 365)

    # dataPred = pd.read_csv(  results_savepath+f"/out0_{iS[item]}_{iE[item]}", dtype=np.float32, header=None).values
    # dataPred = np.expand_dims(dataPred, axis=-1)
    #
    #
    #
    #
    # evaDict = [stat.statError(dataPred[:,365:,0], selected_obs[:,365:])]
    # evaDictLst = evaDict
    # keyLst = ['NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
    # dataBox = list()
    # for iS in range(len(keyLst)):
    #     statStr = keyLst[iS]
    #     temp = list()
    #     for k in range(len(evaDictLst)):
    #         data = evaDictLst[k][statStr]
    #         #data = data[~np.isnan(data)]
    #         temp.append(data)
    #     dataBox.append(temp)
    #
    #
    # print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
    #       np.nanmedian(dataBox[0][0]),
    #       np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
    #       np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))
