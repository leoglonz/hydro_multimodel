import sys

sys.path.append('../../')

from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import default
from hydroDL.model import rnn, crit, train
from hydroDL.master import loadModel
import os
import numpy as np
import pandas as pd
import torch
import random
import json
import datetime as dt
from hydroDL.model import crit
import logging
from tqdm import tqdm
import torch.multiprocessing as mp

# fix the random seeds for reproducibility
random_seed = 111111
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# GPU setting
# which GPU to use when having multiple
train_gpu_id = 1
torch.cuda.set_device(train_gpu_id)

# Setting training options here
PUOpt = 0
buffOpt = 0
TDOpt = False
forType = 'daymet'
buffOptOri = 0
attrnewLst = ['p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
                  'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
                  'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
                  'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                  'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
                  'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']

# Set hyperparameters
EPOCH = 50  # total epochs to train the mode
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
saveEPOCH = 10
T_eval = [19801001, 19951001]
T_eval_lst = utils.time.tRange2Array(T_eval)
Tinv = [19801001, 19951001]

lossFun = crit.RmseLoss()
if forType == 'daymet':
        varF = ['prcp', 'tmean']
        varFInv = ['prcp', 'tmean']
else:
    varF = ['prcp', 'tmax']  # For CAMELS maurer and nldas forcings, tmax is actually tmean
    varFInv = ['prcp', 'tmax']

#def normalize_data(forcInvUN, varFInv, PETInvUN, attrnewLst, attrsUN):
def normalize_data(basin):
    # process data, do normalization and remove nan
    # series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
    # seriesvarLst = varFInv + ['pet']
    # # calculate statistics for normalization and saved to a dictionary
    # statDict = camels.getStatDic(attrLst=attrnewLst, attrdata=attrsUN, seriesLst=seriesvarLst, seriesdata=series_inv)
    # # normalize
    # attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
    # attr_norm[np.isnan(attr_norm)] = 0.0
    # series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
    # series_norm[np.isnan(series_norm)] = 0.0
    series_norm = np.load("/data/yxs275/NROdeSolver/CAMELSData/Camels_inputs_norm.npy" )

    attr_norm = np.load("/data/yxs275/NROdeSolver/CAMELSData/Camels_attr_norm.npy" )

    return series_norm[basin:basin+1,:,:], attr_norm[basin:basin+1,:]


def load_dpl(basin, TinvLoad, EvalLS, varFInv, attrnewLst, gageinfo, T_eval_load, EvalInd):
    df_eval = camels.DataframeCamels(tRange=T_eval_load, subset=EvalLS[basin:basin+1], forType=forType)

    obsUN = df_eval.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
    areas = gageinfo['area'][EvalInd[basin:basin+1]]  # unit km2
    temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1], 1))
    obsEvalUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3  # transform to mm/day

    dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=EvalLS[basin:basin+1], forType=forType)
    forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False,
                                rmNan=False)  # Unit transformation, discharge obs from ft3/s to mm/day
    attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

    return obsEvalUN, forcInvUN, attrsUN, temparea, df_eval


def process_basin(basin, window_size, eval_batch, save_directory, eval_model, BUFFTIME):
    # Define root directory of database and saved output dir
    # Modify this based on your own location of CAMELS dataset and saved models
    rootDatabase = "/data/yxs275/DPL_HBV/"
      # CAMELS dataset root directory
    camels.initcamels(
        rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

    # CAMLES basin info
    gageinfo = camels.gageDict
    hucinfo = gageinfo['huc']
    gageid = gageinfo['id']
    gageidLst = gageid.tolist()

    # same as training, load data based on ALL, PUB, PUR scenarios
    if PUOpt == 0:  # for All the basins
        EvalLS = gageidLst
        puN = 'ALL'
        EvalInd = [gageidLst.index(j) for j in EvalLS]

    elif PUOpt == 1:  # for PUB
        puN = 'PUB'
        # load the subset ID
        # splitPath saves the basin ID of random groups
        splitPath = 'PUBsplitLst.txt'
        with open(splitPath, 'r') as fp:
            testIDLst = json.load(fp)

    elif PUOpt == 2:  # for PUR
        puN = 'ALL'
        # Divide CAMELS dataset into 7 PUR regions
        # get the id list of each region
        regionID = list()
        regionNum = list()
        regionDivide = [[1, 2], [3, 6], [4, 5, 7], [9, 10], [8, 11, 12, 13], [14, 15, 16, 18], [17]]  # seven regions
        for ii in range(len(regionDivide)):
            tempcomb = regionDivide[ii]
            tempregid = list()
            for ih in tempcomb:
                tempid = gageid[hucinfo == ih].tolist()
                tempregid = tempregid + tempid
            regionID.append(tempregid)
            regionNum.append(len(tempregid))
        tarIDLst = regionID
    buffOpt = 0

    T_eval_load = T_eval
    TinvLoad = Tinv
    k_list = []
    obsEvalUN, forcInvUN, attrsUN, temparea, df_eval = load_dpl(basin, TinvLoad, EvalLS, varFInv, attrnewLst,
                                                                gageinfo, T_eval_load, EvalInd)
    # prepare the inputs
    forcUN = df_eval.getDataTs(varLst=varF, doNorm=False, rmNan=False)
    P_input = forcUN[:, :, 0]


    forcUN = torch.tensor(forcUN, dtype=torch.float32)

    O_input = obsEvalUN
    num_epochs = 10
    buffOpt = 0
    # Modify this as the directory where you put PET
    PETDir = rootDatabase + '/pet_harg/' + forType + '/'
    varLstNL = ['PEVAP']
    usgsIdLst = gageid
    if forType == 'maurer':
        tPETRange = [19800101, 20090101]
    else:
        tPETRange = [19800101, 20150101]
    tPETLst = utils.time.tRange2Array(tPETRange)
    ntime = len(tPETLst)
    PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])

    dataTemp = camels.readcsvGage(PETDir, usgsIdLst[basin], varLstNL, ntime)
    PETfull[basin, :, :] = dataTemp
    T_eval_Lst = utils.time.tRange2Array(T_eval_load)
    TinvLst = utils.time.tRange2Array(TinvLoad)
    C, ind1, ind2 = np.intersect1d(T_eval_Lst, tPETLst, return_indices=True)
    PETUN = PETfull[:, ind2, :]
    PETUN = PETUN[EvalInd[basin:basin+1], :, :]  # select basins
    PETUN_tensor = torch.tensor(PETUN, dtype=torch.float32)
    C, ind1, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)

    PETInvUN = PETfull[:, ind2inv, :]
    PETInvUN = PETInvUN[EvalInd[basin:basin+1], :, :]
    # Ensure the directory exists
    #series_norm, attr_norm = normalize_data(forcInvUN, varFInv, PETInvUN, attrnewLst, attrsUN)
    series_norm, attr_norm = normalize_data(basin)
    print("Start multiprcessing")
    # Prepare the arguments for each basin
    zEval = series_norm
    # load potential ET calculated by hargreaves method
    if torch.cuda.is_available():
        # torch.cuda.set_device(0)  # Replace 'device_id' with the appropriate device ID
        torch.cuda.init()

    for i in range(P_input.shape[1] - window_size + 1):
        print(f"Start ::  basin: {basin} :: Day number : {i}")
        k = torch.tensor([1.0], requires_grad=True)  # Initialize k for the current window
        adjusted_P_input = k * torch.tensor(P_input[:, i:i + window_size], dtype=torch.float32)
        # adjusted_P_input = [k * torch.tensor(P_input[i + j]) for j in range(window_size)]

        # adjusted_P_input_numpy = adjusted_P_input.detach().numpy()
        # Initialize optimizer for the current
        optimizer = torch.optim.Adam([k], lr=0.05)
        for _ in range(num_epochs):
            print(f"Basin: {basin} :: Day number:{i} :: epoch number: {_}")
            optimizer.zero_grad()

            forcUN[:, i:i + window_size, 0] = adjusted_P_input
            # forcUN[:, i+1:i+window_size, 0] = p_window
            xEval = torch.cat([forcUN, PETUN_tensor], dim=2)  # new

            xEval[torch.isnan(xEval)] = 0.0  # new

            if buffOpt == 1:  # repeat the first year warm up the first year itself
                zEvalIn = np.concatenate([zEval[:, 0:BUFFTIME, :], zEval], axis=1)
                xEvalIn = np.concatenate([xEval[:, 0:BUFFTIME, :], xEval],
                                         axis=1)  # repeat forcing to warm up the first year
            else:  # no repeat, original data, the first year data would only be used as warmup for the next following year
                zEvalIn = zEval
                xEvalIn = xEval

            attrs = attr_norm

            EvalBuff = xEvalIn.shape[1]
            # filePathLst = master.master.namePred(
            #     eval_out, T_eval, 'All_Buff' + str(EvalBuff), epoch=eval_epoch, targLst=['Qr', 'Q0', 'Q1', 'Q2', 'ET'])
            xEvalIn[torch.isnan(xEvalIn)] = 0.0

            cTemp = np.repeat(
                np.reshape(attrs, [attrs.shape[0], 1, attrs.shape[-1]]), zEvalIn.shape[1], axis=1)
            zEvalIn = np.concatenate([zEvalIn, cTemp], 2)  # Add attributes to historical forcings as the inversion part
            xEvalIn = xEvalIn[:, i:i + window_size + 15, :]
            zEvalIn = zEvalIn[:, i:i + window_size + 15, :]
            evalTuple = (xEvalIn,
                         zEvalIn)  # xTest: input forcings to HBV; zEval: inputs to gA LSTM to learn parameters
            print(f"Start getting output from testModel_eval, basin: {basin} :: day number: {i} :: epoch number : {_}")
            predict_eval = train.testModel_eval(
                eval_model, evalTuple, c=None, batchSize=eval_batch, filePathLst=None)

            print(f"End getting output from testModel_eval,basin:{basin} :: day number: {i} :: epoch number : {_}")

            predict_eval = predict_eval[:1, :window_size, :]

            # predict_eval.requires_grad_(True)
            streamflow_observed = torch.tensor(O_input[:, i:i + window_size, :], dtype=torch.float32)
            loss = lossFun(predict_eval.float(), streamflow_observed.float())
            print(f"Calculated loss: {loss}, day number: {i} epoch number : {_}")

            # Back propagate and adjust k based on the loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # print(f'Epoch {num_epochs}, Window {i}, Gradient of k: {k.grad.item()}')
            del predict_eval
            torch.cuda.empty_cache()
        k_list.append(k.item())
        torch.cuda.empty_cache()
    # Save k values for the current basin to a CSV file
    basin_file_path = os.path.join(save_directory, f'k_values_basin_{basin}.csv')
    pd.DataFrame({'k_value': k_list}).to_csv(basin_file_path, index=False)


def create_batches(num_basins, batch_size):
    return [range(i, min(i + batch_size, num_basins)) for i in range(0, num_basins, batch_size)]


def main():
    Nfea = 12  # number of HBV parameters. 12:original HBV; 13:includes the added dynamic ET para when setting ETMod=True
    BUFFTIME = 365  # for each training sample, to use BUFFTIME days to warm up the states.
    Nmul = 16  # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
    TDOpt = True
    if TDOpt is True:
        tdRep = [1, 13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
        tdRepS = [str(ix) for ix in tdRep]
        # ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
        # Must set below ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
        # If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
        ETMod = True
        Nfea = 13  # should be 13 when setting ETMod=True. 12 when ETMod=False
        dydrop = 0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
        staind = -1  # which time step to use from the learned para time series for those static parameters
        TDN = '/TDTestforc/' + 'TD' + "_".join(tdRepS) + '/'
    else:
        TDN = '/Testforc/'

    eval_batch = 1  # forward number of "eval_batch" basins each time to save GPU memory. You can set this even smaller to save more.
    eval_epoch = 50

    eval_seed = 111111

    puN = 'ALL'
    window_size = 3  # Number of days to consider for each adjustment
    # this testsave_path should be consistent with where you save your model
    eval_save_path = 'CAMELSDemo/dPLHBV/' + puN + '/Testforc/' + forType + \
                     '/BuffOpt' + str(buffOptOri) + '/RMSE_para0.25' + '/' + str(eval_seed)

    foldstr = 'Fold' + str(1)
    exp_info = 'T_' + str(T_eval[0]) + '_' + str(T_eval[1]) + '_BS_' + str(BATCH_SIZE) + '_HS_' + str(HIDDENSIZE) \
               + '_RHO_' + str(RHO) + '_NF_' + str(Nfea) + '_Buff_' + str(BUFFTIME) + '_Mul_' + str(Nmul)
    rootOut = "/data/yxs275/DPL_HBV/output/record_16/"  # Update this path as per your preference
    # the final path to test with the trained model saved in


    runBUFF = 0
  #  eval_out = os.path.join(rootOut, eval_save_path, foldstr, exp_info)

    eval_out ="/data/yxs275/DPL_HBV/output/record_16/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymet/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/"

    eval_model = loadModel(eval_out, epoch=eval_epoch)
    eval_model.inittime = runBUFF
    eval_model.dydrop = 0.0
    # load potential ET calculated by hargreaves method
    save_directory = '/data/yxs275/DPL_HBV/DA/k_per_basin/'
    os.makedirs(save_directory, exist_ok=True)
    # Setup the multiprocessing method to spawn (recommended for CUDA)
    mp.set_start_method('spawn', force=True)
    # Run the process_basin function in parallel for each basin
    # with mp.Pool() as pool:
    # pool.starmap(process_basin, basin_args)
    # Run the process_basin function in parallel for each basin
    # Prepare batches of basins
    num_basins = 671
    batch_size = 10
    num_epochs = 10
    basin_batches = create_batches(num_basins, batch_size)
    for batch in basin_batches:
        for basin in batch:
            process_basin(basin, window_size, eval_batch, save_directory, eval_model, BUFFTIME)
        # with mp.Pool() as pool:
        #     pool.starmap(process_basin, [(
        #         basin, window_size, eval_batch, save_directory, eval_model, BUFFTIME) for basin in batch])


if __name__ == "__main__":
    main()
