import sys
sys.path.append('../../')

from hydroDL.model import train
from hydroDL.post import plot, stat
import os
import numpy as np
import torch
import random
import pandas as pd
import json
import multiprocessing
import glob
import time

## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Path of model
Model_path =  "/projects/mhpi/yxs275/model/" + '/dPL_local_daymet_new_attr/'
out = os.path.join(Model_path, "exp_EPOCH50_BS100_RHO365_HS512_trainBuff365")

#Path of data
#data_folder = "/data/yxs275/CONUS_data/HUC10/daymet_format/validation/"
data_folder = "/projects/mhpi/yxs275/Data/conus_merit_1000/"
##epoch of model to use
testepoch = 50

## ids of GPUs to use
GPU_ids = [2,3,4,5,6]


## number of GPUs to use
num_gpus = len(GPU_ids)

## Path to save the results
results_savepath = "/projects/mhpi/yxs275/DM_output/" + '/dPL_local_daymet_filled_NaN_merit_forward/'
if os.path.exists(results_savepath) is False:
    os.mkdir(results_savepath)

## Pick the regions/groups for forwarding
attributefileLst = glob.glob(data_folder+"attr/region*.csv")
forcingfileLst = glob.glob(data_folder+"forcing/region*.npy")
attributefileLst.sort()
forcingfileLst.sort()




def forward_on_gpu(input):
    item, gpu_id = input

    ## load attribute file
    attributeBatch_file = attributefileLst[item]
    region_number = attributeBatch_file.split("region_")[-1].split(".csv")[0]
    print("GPU ", gpu_id, "is working on region ", region_number)

    attributeBatch_df = pd.read_csv(attributeBatch_file)
    attributeBatchLst = attributeBatch_df.columns[1:]

    attributeBatch = attributeBatch_df.values[:,1:]

    #load forcing
    xTrain  =  np.load(data_folder + f"forcing/region_{region_number}.npy")
    xTrain = fill_Nan(xTrain)

    log_norm_cols=['prcp', 'pr', 'total_precipitation', 'pre',  'LE',
                       'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'susm', 'smp', 'ssma', 'susma',
                       'usgsFlow', 'streamflow', 'qobs']
    attributeLst = ['catchsize', 'ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
                    'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
                    'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
                    'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
                    'meanelevation', 'meanslope', 'permafrost', 'permeability',
                    'seasonality_P', 'seasonality_PET', 'snow_fraction',
                    'snowfall_fraction']

    ## Select the attributes used in the model

    [C, ind1, SubInd] = np.intersect1d(attributeLst, attributeBatchLst, return_indices=True)
    sorted_SubInd = [x for x, _ in sorted(zip(SubInd, ind1), key=lambda pair: pair[1])]
    attribute = attributeBatch[:, sorted_SubInd]


    ## Input normalization
    with open(out + '/dapengscaler_stat.json') as f:
        stat_dict = json.load(f)
    stat_dict["catchsize"] = stat_dict["area"]
    forcing_LSTM_norm = _trans_norm(
        xTrain.copy(), ['prcp','tmean','pet'], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
    )
    forcing_LSTM_norm [forcing_LSTM_norm != forcing_LSTM_norm] = 0
    xTrain[xTrain != xTrain] = 0
    attribute_norm = _trans_norm(attribute, list(attributeLst), stat_dict, log_norm_cols=log_norm_cols, to_norm=True)
    attribute_norm[attribute_norm != attribute_norm] = 0
    attribute_norm = np.expand_dims(attribute_norm, axis=1)
    attribute_norm = np.repeat(attribute_norm, forcing_LSTM_norm.shape[1], axis=1)

    # load the model

    model_path = out
    print("Load model from ", model_path)


    zTest = np.concatenate([forcing_LSTM_norm, attribute_norm], 2)  # Add attributes to historical forcings as the inversion part
    xTest = xTrain
    testTuple = (xTest[:,:,:], zTest[:,:,:])
    testbatch =80 #len(indexes)

    filePathLst = [results_savepath+f"/Qr_{region_number}.npy",results_savepath+f"/Q0_{region_number}",results_savepath+f"/Q1_{region_number}",results_savepath+f"/Q2_{region_number}",results_savepath+f"/ET_{region_number}"]
    device = torch.device("cuda:" + str(gpu_id))

    modelFile = os.path.join(model_path, 'model_Ep' + str(testepoch) + '.pt')
    testmodel = torch.load(modelFile, map_location=device)

    testmodel.inittime = 0


    train.testModel_multiGPU(
        testmodel, testTuple, c=None, device =  device, batchSize=testbatch, filePathLst=filePathLst)
    print(f"Group/region {item} is done on GPU {gpu_id}")

    if region_number =="validation":
        dataPred = np.load(results_savepath+f"/Qr_{region_number}.npy")
        streamflow = np.expand_dims(np.load(data_folder + f"/streamflow_{region_number}.npy"),axis =-1)
        basin_area = np.expand_dims(attributeBatch_df["catchsize"].values, axis=1)
        streamflow_trans = _basin_norm(
            streamflow[:, :, 0:  1].copy(), basin_area, to_norm=True
        )

        evaDict = [stat.statError(dataPred[:, 365:], streamflow_trans[:, 365:, 0])]
        evaDictLst = evaDict
        keyLst = ['NSE', 'KGE', 'FLV', 'FHV', 'lowRMSE', 'highRMSE', 'rdMax', 'absFLV', 'absFHV']
        dataBox = list()
        for iS in range(len(keyLst)):
            statStr = keyLst[iS]
            temp = list()
            for k in range(len(evaDictLst)):
                data = evaDictLst[k][statStr]
                # data = data[~np.isnan(data)]
                temp.append(data)
            dataBox.append(temp)

        print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
              np.nanmedian(dataBox[0][0]),
              np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
              np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]),
              np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))


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

        flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

    else:

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

#main parallel code

startTime = time.time()
#forward_on_gpu((0,0))
items = [x for x in range(len(attributefileLst))]
GPU_ids_list  = [GPU_ids[x % len(GPU_ids)] for x in items]

pool = multiprocessing.Pool(processes=num_gpus*2)


results = pool.imap_unordered(forward_on_gpu, ((idx, gpuid) for idx, gpuid in enumerate(GPU_ids_list)))

pool.close()
pool.join()

print("Cost time: ", time.time() - startTime  )
