"""
Training and Testing functions for the hydro models (PGML versions + HydroDL)
"""

import os
import platform

import numpy as np
import pandas as pd
import torch
import json
import datetime as dt
from tqdm import tqdm

from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master.master import loadModel
from hydroDL.model import train

from core.utils.randomseed_config import randomseed_config
from core.utils.master import create_output_dirs
from MODELS.loss_functions.get_loss_function import get_lossFun
from core.data_processing.data_loading import loadData
from core.data_processing.normalization import transNorm
from core.data_processing.model import (
    take_sample_test,
    converting_flow_from_ft3_per_sec_to_mm_per_day
)

import time
from tqdm import tqdm
from hydroDL.model_new import crit
from core.data_processing.data_prep import selectSubset, randomIndex



def test_differentiable_model(args, diff_model):
    """
    This function collects and outputs the model predictions and the corresponding
    observations needed to run statistical analyses.

    If rerunning testing in a Jupyter environment, you will need to re-import args
    as `batch_size` is overwritten in this function and will throw an error if the
    overwrite is attempted a second time.
    """
    warm_up = args["warm_up"]
    nmul = args["nmul"]
    diff_model.eval()
    # read data for test time range
    dataset_dictionary = loadData(args, trange=args["t_test"])
    np.save(os.path.join(args["out_dir"], "x.npy"), dataset_dictionary["x_NN"])  # saves with the overlap in the beginning
    # normalizing
    x_NN_scaled = transNorm(args, dataset_dictionary["x_NN"], varLst=args["varT_NN"], toNorm=True)
    c_NN_scaled = transNorm(args, dataset_dictionary["c_NN"], varLst=args["varC_NN"], toNorm=True)
    c_NN_scaled = np.repeat(np.expand_dims(c_NN_scaled, 0), x_NN_scaled.shape[0], axis=0)
    dataset_dictionary["inputs_NN_scaled"] = np.concatenate((x_NN_scaled, c_NN_scaled), axis=2)
    del x_NN_scaled, dataset_dictionary["x_NN"]
    # converting the numpy arrays to torch tensors:
    for key in dataset_dictionary.keys():
        dataset_dictionary[key] = torch.from_numpy(dataset_dictionary[key]).float()

    # args_mod = args.copy()
    args["batch_size"] = args["no_basins"]
    nt, ngrid, nx = dataset_dictionary["inputs_NN_scaled"].shape

    # Making lists of the start and end indices of the basins for each batch.
    batch_size = args["batch_size"]
    iS = np.arange(0, ngrid, batch_size)    # Start index list.
    iE = np.append(iS[1:], ngrid)   # End.

    list_out_diff_model = []
    for i in tqdm(range(0, len(iS)), unit='Batch'):
        dataset_dictionary_sample = take_sample_test(args, dataset_dictionary, iS[i], iE[i])

        out_diff_model = diff_model(dataset_dictionary_sample)
        # Convert all tensors in the dictionary to CPU
        out_diff_model_cpu = {key: tensor.cpu().detach() for key, tensor in out_diff_model.items()}
        # out_diff_model_cpu = tuple(outs.cpu().detach() for outs in out_diff_model)
        list_out_diff_model.append(out_diff_model_cpu)

    # getting rid of warm-up period in observation dataset and making the dimension similar to
    # converting numpy to tensor
    # y_obs = torch.tensor(np.swapaxes(y_obs[:, warm_up:, :], 0, 1), dtype=torch.float32)
    # c_hydro_model = torch.tensor(c_hydro_model, dtype=torch.float32)
    y_obs = converting_flow_from_ft3_per_sec_to_mm_per_day(args,
                                                           dataset_dictionary["c_NN"],
                                                           dataset_dictionary["obs"][warm_up:, :, :])

    return list_out_diff_model, y_obs



def test_dp_hbv(rootdir):
    """
    This script runs all of the necessary data cleaning, along with the training regement, for Dapeng's dynamic dPLHBV model.
    """
    ## GPU setting (check m ac vs windows), and
    ## Define root directory of database and saved output dir.
    forType = 'daymet'
    flow_regime = 0  # 1 is high flow expert.
    
    if platform.system() == 'Darwin':
        # Use mac M1 GPUs
        if torch.backends.mps.is_available():
            # device = torch.device('mps')
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        print("Using device", device)

        # Setting dirs
        rootDatabase = os.path.join(os.path.sep, '/Users/leoglonz/Desktop/water/data', 'Camels')  # CAMELS dataset root directory

        rootOut = os.path.join(os.path.sep, '/Users/leoglonz/Desktop/water/data/model_runs', 'rnnStreamflow')  # Model output root directory

    elif platform.system() == 'Windows':
        # Use nvidia GPU
        testgpuid = 0
        torch.cuda.set_device(testgpuid)

        # Setting dirs
        rootDatabase = os.path.join(os.path.sep, 'D:\\', 'code_repos', 'water', 'data', 'Camels')  # CAMELS dataset root directory

        rootOut = os.path.join(os.path.sep, 'D:\\', 'code_repos', 'water', 'data', 'model_runs', 'rnnStreamflow')  # Model output root directory

    elif platform.system() == 'Linux':
        # Use nvidia GPU in Colab
        testgpuid = 0
        torch.cuda.set_device(testgpuid)

        # Setting dirs
        rootDatabase = '/content/Camels' # CAMELS dataset root directory

        rootOut = os.path.join(os.path.sep, '/content/drive/MyDrive/Colab/data/model_runs', 'rnnStreamflow')  # Model output root directory

    else:
        raise ValueError('Unsupported operating system.')

    camels.initcamels(flow_regime, forType=forType,rootDB=rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

    ## setting options, keep the same as your training
    PUOpt = 0  # 0 for All; 1 for PUB; 2 for PUR;
    buffOptOri = 0  # original buffOpt, Same as what you set for training
    buffOpt = 1  # control load training data 0: do nothing; 1: repeat first year; 2: load one more year

    ## Hyperparameters, keep the same as your training setup
    BATCH_SIZE = 100
    RHO = 365
    HIDDENSIZE = 256
    Ttrain = [19801001, 19951001]  # Training period
    # Ttrain = [19891001, 19991001]  # PUB/PUR period
    Tinv = [19801001, 19951001] # dPL Inversion period
    # Tinv = [19891001, 19991001]  # PUB/PUR period
    Nfea = 13 # number of HBV parameters, 13 includes the added one for ET eq
    BUFFTIME = 365
    routing = True
    Nmul = 16
    comprout = False
    compwts = False
    pcorr = None

    tdRep = [1, 13] # index of dynamic parameters
    tdRepS = [str(ix) for ix in tdRep]
    dydrop = 0.0 # the possibility to make dynamic become static; 0.0, all dynamic; 1.0, all static
    staind = -1

    ## Testing parameters
    # Ttest = [19951001, 20101001]  # testing period
    Ttest = Ttrain
    # Ttest = [19891001, 19991001]  # PUB/PUR period
    # TtestLst = utils.time.tRange2Array(Ttest)
    TtestLst = utils.time.tRange2Array(Ttrain)
    # TtestLoad = [19951001, 20101001]  # could potentially use this to load more forcings before testing period as warm-up
    TtestLoad = [19801001, 19951001]
    # TtestLoad = [19801001, 19991001]  # PUB/PUR period
    testbatch = 30  # forward number of "testbatch" basins each time to save GPU memory. You can set this even smaller to save more.
    testepoch = 50
    testseed = 111111


    # CAMLES basin info
    gageinfo = camels.gageDict
    hucinfo = gageinfo['huc']
    gageid = gageinfo['id']
    gageidLst = gageid.tolist()

    # same as training, load data based on ALL, PUB, PUR scenarios
    if PUOpt == 0: # for All the basins
        puN = 'ALL'
        tarIDLst = [gageidLst]

    elif PUOpt == 1: # for PUB
        puN = 'PUB'
        # load the subset ID
        # splitPath saves the basin ID of random groups
        splitPath = 'PUBsplitLst.txt'
        with open(splitPath, 'r') as fp:
            testIDLst=json.load(fp)
        tarIDLst = testIDLst

    elif PUOpt == 2: # for PUR
        puN = 'PUR'
        # Divide CAMELS dataset into 7 PUR regions
        # get the id list of each region
        regionID = list()
        regionNum = list()
        regionDivide = [ [1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17] ] # seven regions
        for ii in range(len(regionDivide)):
            tempcomb = regionDivide[ii]
            tempregid = list()
            for ih in tempcomb:
                tempid = gageid[hucinfo==ih].tolist()
                tempregid = tempregid + tempid
            regionID.append(tempregid)
            regionNum.append(len(tempregid))
        tarIDLst = regionID

    # define the matrix to save results
    predtestALL = np.full([len(gageid), len(TtestLst), 5], np.nan)
    obstestALL = np.full([len(gageid), len(TtestLst), 1], np.nan)

    # this testsave_path should be consistent with where you save your model
    testsave_path = 'CAMELSDemo/dPLHBV/' + puN + '/TDTestforc/' + 'TD'+"_".join(tdRepS)+'/' + forType + \
                    '/BuffOpt' + str(buffOptOri) + '/RMSE_para0.25'+'/'+str(testseed)
    ## load data and test the model
    nstart = 0
    logtestIDLst = []
    # loop to test all the folds for PUB and PUR. The default is you need to have run all folds, but if
    # you only run one fold for PUB or PUR and just want to test that fold (i.e. fold X), you may set this as:
    # for ifold in range(X, X+1):
    for ifold in range(1, len(tarIDLst)+1):
        testfold = ifold
        TestLS = tarIDLst[testfold - 1]
        TestInd = [gageidLst.index(j) for j in TestLS]
        if PUOpt == 0:  # Train and test on ALL basins
            TrainLS = gageidLst
            TrainInd = [gageidLst.index(j) for j in TrainLS]
        else:
            TrainLS = list(set(gageid.tolist()) - set(TestLS))
            TrainInd = [gageidLst.index(j) for j in TrainLS]
        gageDic = {'TrainID':TrainLS, 'TestID':TestLS}

        nbasin = len(TestLS) # number of basins for testing

        # get the dir path of the saved model for testing
        foldstr = 'Fold' + str(testfold)
        exp_info = 'T_'+str(Ttrain[0])+'_'+str(Ttrain[1])+'_BS_'+str(BATCH_SIZE)+'_HS_'+str(HIDDENSIZE)\
                +'_RHO_'+str(RHO)+'_NF_'+str(Nfea)+'_Buff_'+str(BUFFTIME)+'_Mul_'+str(Nmul)
        # the final path to test with the trained model saved in
        testout = os.path.join(rootOut, testsave_path, foldstr, exp_info)
        testmodel = loadModel(testout, epoch=testepoch)

        # apply buffOpt for loading the training data
        if buffOpt == 2: # load more "BUFFTIME" forcing before the training period
            sd = utils.time.t2dt(Ttrain[0]) - dt.timedelta(days=BUFFTIME)
            sdint = int(sd.strftime("%Y%m%d"))
            TtrainLoad = [sdint, Ttrain[1]]
            TinvLoad = [sdint, Ttrain[1]]
        else:
            TtrainLoad = Ttrain
            TinvLoad = Tinv

        # prepare input data
        # load camels dataset
        if forType == 'daymet':
            varF = ['prcp', 'tmean']
            varFInv = ['prcp', 'tmean']
        else:
            varF = ['prcp', 'tmax']  # tmax is tmean here for the original CAMELS maurer and nldas forcing
            varFInv = ['prcp', 'tmax']

        attrnewLst = [ 'p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
                    'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
                    'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
                    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                    'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
                    'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']


        # for HBV training inputs
        dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=forType)
        forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False, flow_regime=flow_regime)
        # obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)  # useless for testing

        # for dPL inversion training data
        dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=forType)
        forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False, flow_regime=flow_regime)
        attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False, flow_regime=flow_regime)

        # for HBV testing input
        dfTest = camels.DataframeCamels(tRange=TtestLoad, subset=TestLS, forType=forType)
        forcTestUN = dfTest.getDataTs(varLst=varF, doNorm=False, rmNan=False, flow_regime=flow_regime)
        obsTestUN = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False, flow_regime=flow_regime)
        attrsTestUN = dfTest.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False, flow_regime=flow_regime)

        # Transform obs from ft3/s to mm/day
        # areas = gageinfo['area'][TrainInd] # unit km2
        # temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1],1))
        # obsUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3  # useless for testing
        areas = gageinfo['area'][TestInd] # unit km2
        temparea = np.tile(areas[:, None, None], (1, obsTestUN.shape[1],1))
        obsTestUN = (obsTestUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3 # transform to mm/day

        # load potential ET calculated by hargreaves method
        varLstNL = ['PEVAP']
        usgsIdLst = gageid
        if forType == 'maurer':
            tPETRange = [19800101, 20090101]
        else:
            tPETRange = [19800101, 20150101]
        tPETLst = utils.time.tRange2Array(tPETRange)
        PETDir = rootDatabase + '/pet_harg/' + forType + '/'
        ntime = len(tPETLst)
        PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
        for k in range(len(usgsIdLst)):
            dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
            PETfull[k, :, :] = dataTemp

        TtrainLst = utils.time.tRange2Array(TtrainLoad)
        TinvLst = utils.time.tRange2Array(TinvLoad)
        TtestLoadLst = utils.time.tRange2Array(TtestLoad)
        C, ind1, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
        PETUN = PETfull[:, ind2, :]
        PETUN = PETUN[TrainInd, :, :] # select basins
        C, ind1, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
        PETInvUN = PETfull[:, ind2inv, :]
        PETInvUN = PETInvUN[TrainInd, :, :]
        C, ind1, ind2test = np.intersect1d(TtestLoadLst, tPETLst, return_indices=True)
        PETTestUN = PETfull[:, ind2test, :]
        PETTestUN = PETTestUN[TestInd, :, :]

        # process data, do normalization and remove nan
        series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
        seriesvarLst = varFInv + ['pet']
        # load the saved statistics
        statFile = os.path.join(testout, 'statDict.json')
        with open(statFile, 'r') as fp:
            statDict = json.load(fp)

        # normalize
        attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True, flow_regime=flow_regime)
        attr_norm[np.isnan(attr_norm)] = 0.0
        series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True, flow_regime=flow_regime)
        series_norm[np.isnan(series_norm)] = 0.0

        attrtest_norm = camels.transNormbyDic(attrsTestUN, attrnewLst, statDict, toNorm=True, flow_regime=flow_regime)
        attrtest_norm[np.isnan(attrtest_norm)] = 0.0
        seriestest_inv = np.concatenate([forcTestUN, PETTestUN], axis=2)
        seriestest_norm = camels.transNormbyDic(seriestest_inv, seriesvarLst, statDict, toNorm=True, flow_regime=flow_regime)
        seriestest_norm[np.isnan(seriestest_norm)] = 0.0

        # prepare the inputs
        zTrain = series_norm
        xTrain = np.concatenate([forcUN, PETUN], axis=2) # HBV forcing
        xTrain[np.isnan(xTrain)] = 0.0

        if buffOpt == 1: # repeat the first year for buff
            zTrainIn = np.concatenate([zTrain[:,0:BUFFTIME,:], zTrain], axis=1)
            xTrainIn = np.concatenate([xTrain[:,0:BUFFTIME,:], xTrain], axis=1) # Bufftime for the first year
            # yTrainIn = np.concatenate([obsUN[:,0:BUFFTIME,:], obsUN], axis=1)
        else: # no repeat, original data
            zTrainIn = zTrain
            xTrainIn = xTrain
            # yTrainIn = obsUN

        forcTuple = (xTrainIn, zTrainIn)
        attrs = attr_norm

        ## Prepare the testing data and forward the trained model for testing
        runBUFF = 0

        # TestBuff = 365 # Use 365 days forcing to warm up the model for testing
        TestBuff = xTrain.shape[1]  # Use the whole training period to warm up the model for testing
        # TestBuff = len(TtestLoadLst) - len(TtestLst)  # use the redundantly loaded data to warm up

        testmodel.inittime = runBUFF
        testmodel.dydrop = 0.0

        # prepare file name to save the testing predictions
        filePathLst = master.master.namePred(
            testout, Ttest, 'All_Buff'+str(TestBuff), epoch=testepoch, targLst=['Qr', 'Q0', 'Q1', 'Q2', 'ET'])

        # prepare the inputs for TESTING
        if PUOpt == 0: # for ALL basins, temporal generalization test
            testmodel.staind = TestBuff-1
            zTest = np.concatenate([series_norm[:, -TestBuff:, :], seriestest_norm], axis=1)
            xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)  # HBV forcing
            # forcings to warm up the model. Here use the forcing of training period to warm up
            xTestBuff = xTrain[:, -TestBuff:, :]
            xTest = np.concatenate([xTestBuff, xTest], axis=1)
            obs = obsTestUN[:, 0:, :]  # starts with 0 when not loading more data before testing period

        else:  # for PUB/PUR cases, different testing basins. Load more forcings to warm up.
            # testmodel.staind = -1
            testmodel.staind = TestBuff-1
            zTest = seriestest_norm
            xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)  # HBV forcing
            obs = obsTestUN[:, TestBuff:, :]  # exclude loaded obs in warming up period (first TestBuff days) for evaluation

        # Final inputs to the test model
        xTest[np.isnan(xTest)] = 0.0
        attrtest = attrtest_norm
        cTemp = np.repeat(
            np.reshape(attrtest, [attrtest.shape[0], 1, attrtest.shape[-1]]), zTest.shape[1], axis=1)
        zTest = np.concatenate([zTest, cTemp], 2) # Add attributes to historical forcings as the inversion part
        testTuple = (xTest, zTest) # xTest: input forcings to HBV; zTest: inputs to gA LSTM to learn parameters

        # forward the model and save results
        train.testModel(
            testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst)

        # read out the saved forward predictions
        dataPred = np.ndarray([obs.shape[0], obs.shape[1]+TestBuff-runBUFF, len(filePathLst)])
        for k in range(len(filePathLst)):
            filePath = filePathLst[k]
            dataPred[:, :, k] = pd.read_csv(
                filePath, dtype=np.float64, header=None).values
        # save the predictions to the big matrix
        predtestALL[nstart:nstart+nbasin, :, :] = dataPred[:, TestBuff-runBUFF:, :]
        obstestALL[nstart:nstart+nbasin, :, :] = obs
        nstart = nstart + nbasin
        logtestIDLst = logtestIDLst + TestLS

    return predtestALL, obstestALL



def test_models(models, args_list, hbv_save_path):
    loss_funcs = dict()
    preds = dict()
    y_obs = dict()

    for mod in models:
        mod = str(mod)

        if mod in ['HBV', 'SACSMA', 'SACSMA_snow', 'marrmot_PRMS']:
            randomseed_config(seed=args_list[mod]["randomseed"][0])
            # Creating output directories and adding them to args.
            args_list[mod] = create_output_dirs(args_list[mod])
            args = args_list[mod]

            loss_funcs[mod] = get_lossFun(args_list[mod])

            # spec = 'LSTM_E' + str(args['EPOCHS']) + '_R' + str(args['rho']) + '_B' + str(args['batch_size']) + '_H' + str(args['hidden_size']) + '_tr1980_1995_n' + str(args['nmul'])
            # modelFile = os.path.join(args["out_dir"], mod, spec)

            modelFile = os.path.join(args["out_dir"], "model_Ep" + str(args['EPOCHS']) + ".pt")
            models[mod] = torch.load(modelFile)     # Append instanced models.

            print("Collecting predictions, observations for %s in batches of %i." %(mod, args['no_basins']))
            preds[mod], y_obs[mod] = test_differentiable_model(args=args,
                                                                    diff_model=models[mod])
        elif mod in ['dPLHBV_stat', 'dPLHBV_dyn']:
            print("Collecting predictions, observations for dPLHBV (HydroDL).")
            preds[mod], y_obs[mod] = test_dp_hbv(hbv_save_path)
        else:
            raise ValueError(f"Unsupported model type in `models`.")
    
    return preds, y_obs, models



def train_ensemble(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               startEpoch=1,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0,
               prcp_loss_factor = 15,
               smooth_loss_factor = 0,
               ):
    """
    inputs:
    x - input
    z - normalized x input
    y - target, or observed values
    c - constant input, attributes
    """

    batchSize, rho = miniBatch
    if type(x) is tuple or type(x) is list:
        x, z = x

    ngrid, nt, nx = x.shape  # ngrid= # basins, nt= # timesteps, nx= # attributes

    if c is not None:
        nx = nx + c.shape[-1]
    if batchSize >= ngrid:
        # Cannot have more batches than there are basins.
        batchSize = ngrid

    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime)))
        )
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(list(model.parameters()))
    model.zero_grad()

    # Save file.
    if saveFolder is not None:
        os.makedirs(saveFolder, exist_ok=True)
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')

    for iEpoch in range(startEpoch, nEpoch + 1):
        lossEp = 0
        loss_prcp_Ep = 0
        loss_sf_Ep = 0
        # loss_smooth_Ep = 0

        t0 = time.time()
        prog_str = "Epoch " + str(iEpoch) + "/" + str(nEpoch)

        for iIter in tqdm(range(0, nIterEp), desc=prog_str, leave=False):
            # training iterations

            iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)

            xTrain = selectSubset(x, iGrid, iT, rho, bufftime=bufftime)
            yTrain = selectSubset(y, iGrid, iT, rho)
            # zTrain = selectSubset(z, iGrid, iT, rho, c=c, bufftime=bufftime)
            zTrain = selectSubset(z, iGrid, iT, rho, bufftime=bufftime)

            # calculate loss and weights `wt`.
            xP, prcp_loss, prcp_weights = model(xTrain, zTrain, prcp_loss_factor)
            yP = torch.sum(xTrain * prcp_weights, dim=2).unsqueeze(2)

            # Consider the buff time for initialization.
            if bufftime > 0:
                yP = yP[bufftime:,:,:]

            # get loss
            if type(lossFun) in [crit.NSELossBatch, crit.NSESqrtLossBatch]:
                loss_sf = lossFun(yP, yTrain, iGrid)
                loss =  loss_sf + prcp_loss
            else:
                loss_sf = lossFun(yP, yTrain)
                loss = loss_sf + prcp_loss

            loss.backward()
            optim.step()
            optim.zero_grad()
            lossEp = lossEp + loss.item()

            try:
                loss_prcp_Ep = loss_prcp_Ep + prcp_loss.item()
            except:
                pass

            loss_sf_Ep = loss_sf_Ep + loss_sf.item()

            # if iIter % 100 == 0:
            #     print('Iter {} of {}: Loss {:.3f}'.format(iIter, nIterEp, loss.item()))

        # print loss
        lossEp = lossEp / nIterEp
        loss_sf_Ep = loss_sf_Ep / nIterEp
        loss_prcp_Ep = loss_prcp_Ep / nIterEp

        logStr = 'Epoch {} Loss {:.3f}, Streamflow Loss {:.3f}, Precipitation Loss {:.3f}, time {:.2f}'.format(
            iEpoch, lossEp, loss_sf_Ep, loss_prcp_Ep,
            time.time() - t0)
        print(logStr)

        # Save model and loss.
        if saveFolder is not None:
            rf.write(logStr + '\n')
            if iEpoch % saveEpoch == 0:
                # save model
                modelFile = os.path.join(saveFolder,
                                         'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)

    if saveFolder is not None:
        rf.close()

    return model



import math
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.autograd as autograd

from hydroDL.model.dropout import DropMask, createMask



class CudnnLstm(nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod="drW", gpu=0, seed=42):
        super(CudnnLstm, self).__init__()
        self.name = 'CudnnLstm'
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()
        self.seed = seed
        self.is_legacy = True

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr, self.seed)
        self.maskW_hh = createMask(self.w_hh, self.dr, self.seed)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        # input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        # self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hiddenSize,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hiddenSize,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]



class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, warmUpDay=None):
        super(CudnnLstmModel, self).__init__()
        self.name = 'CudnnLstmModel'
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr
        )
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.is_legacy = True
        # self.drtest = torch.nn.Dropout(p=0.4)
        self.warmUpDay = warmUpDay

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        :param inputs: a dictionary of input data (x and potentially z data)
        :param doDropMC:
        :param dropoutFalse:
        :return:
        """
        # if not self.warmUpDay is None:
        #     x, warmUpDay = self.extend_day(x, warm_up_day=self.warmUpDay)

        x0 = F.relu(self.linearIn(x))

        outLSTM, (hn, cn) = self.lstm(
            x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse
        )
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)

        # if not self.warmUpDay is None:
        #     out = self.reduce_day(out, warm_up_day=self.warmUpDay)

        return out

    def extend_day(self, x, warm_up_day):
        x_num_day = x.shape[0]
        warm_up_day = min(x_num_day, warm_up_day)
        x_select = x[:warm_up_day, :, :]
        x = torch.cat([x_select, x], dim=0)
        return x, warm_up_day

    def reduce_day(self, x, warm_up_day):
        x = x[warm_up_day:,:,:]
        return x