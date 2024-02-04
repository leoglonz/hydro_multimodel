"""
-------------------------------------
Code pulled and reorganized from `PGML_STemp > HydroModels'
for demonstration purposes.

Credit Rahmani & Song et al.
-------------------------------------

Purpose: This file runs training on an ensemble of PyTorch hydro models for the
15-year train window of 1980/10/01 to 1995/09/30 (or [19801001,19951001]) with 
all 671 CAMELS basins as default.

The desired ensemble method is specified by `ensemble_type`, but, if necessary, 
individual model runs can also be performed by specifying only one model in `models`
and `arg_list`.
"""
from config.read_configurations import config_hbv as hbvArgs
from config.read_configurations import config_prms as prmsArgs
from config.read_configurations import config_sacsma as sacsmaArgs
from config.read_configurations import config_hbv_hydrodl as hbvhyArgs


import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats

from post import plot
from MODELS.test_dp_HBV import test_dp_hbv

from core.utils.randomseed_config import randomseed_config
from core.utils.master import create_output_dirs
from MODELS.loss_functions.get_loss_function import get_lossFun
from core.data_processing.data_loading import loadData
from core.data_processing.normalization import transNorm
from core.data_processing.model import (
    take_sample_test,
    converting_flow_from_ft3_per_sec_to_mm_per_day
)

import warnings
warnings.filterwarnings("ignore")



##-----## Multi-model Parameters ##-----##
# Setting dictionaries to separately manage each diff model's attributes.
models = {'hbvhy': None, 'SACSMA':None, 'marrmot_PRMS':None}  # 'hbv':None
args_list = {'SACSMA': sacsmaArgs, 'marrmot_PRMS':prmsArgs,'hbv': hbvhyArgs} # hbvArgs

ENSEMBLE_TYPE = 'median'  # 'avg', 'max', 'softmax'
OUT_DIR = 'D:\\data\\model_runs\\hydro_multimodel_results\\'


def test_multi_model():
    """
    Runs testing on each model in the ensemble, and then collects together 
    results into array for more efficient data merging between models later on.
    """
    loss_funcs = dict()
    model_output = dict()
    y_obs = dict()

    for mod in models:
        mod = str(mod)

        if mod in ['SACSMA', 'marrmot_PRMS', 'farshid_HBV']:
            randomseed_config(seed=args_list[mod]["randomseed"][0])
            # Creating output directories and adding them to args.
            args_list[mod] = create_output_dirs(args_list[mod])
            args = args_list[mod]

            loss_funcs[mod] = get_lossFun(args_list[mod])

            modelFile = os.path.join(args["out_dir"], "model_Ep" + str(args['EPOCHS']) + ".pt")        
            models[mod] = torch.load(modelFile)     # Append instanced models.

            print("Collecting predictions, observations for %s in batches of %i." %(mod, args['no_basins']))
            model_output[mod], y_obs[mod] = test_differentiable_model(args=args, 
                                                                    diff_model=models[mod])
        elif mod == 'hbv':
            print("Collecting predictions, observations for HBV.")
            model_output[mod], y_obs[mod] = test_dp_hbv()
        else:
            raise ValueError(f"Unsupported model type in `models`.")
        
    return model_output, y_obs


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


def calFDC(data):
    # data = Ngrid * Nday
    Ngrid, Nday = data.shape
    FDC100 = np.full([Ngrid, 100], np.nan)
    for ii in range(Ngrid):
        tempdata0 = data[ii, :]
        tempdata = tempdata0[~np.isnan(tempdata0)]
        # deal with no data case for some gages
        if len(tempdata)==0:
            tempdata = np.full(Nday, 0)
        # sort from large to small
        temp_sort = np.sort(tempdata)[::-1]
        # select 100 quantile points
        Nlen = len(tempdata)
        ind = (np.arange(100)/100*Nlen).astype(int)
        FDCflow = temp_sort[ind]
        if len(FDCflow) != 100:
            raise Exception('unknown assimilation variable')
        else:
            FDC100[ii, :] = FDCflow

    return FDC100


def statError(pred, target):
    """
    Compute performance statistics on a model's predictions vs. target data.
    """
    ngrid, nt = pred.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
    # Bias
        Bias = np.nanmean(pred - target, axis=1)
        # RMSE
        RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
        # ubRMSE
        predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
        targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
        predAnom = pred - predMean
        targetAnom = target - targetMean
        ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
        # FDC metric
        predFDC = calFDC(pred)
        targetFDC = calFDC(target)
        FDCRMSE = np.sqrt(np.nanmean((predFDC - targetFDC) ** 2, axis=1))
    # rho R2 NSE
        Corr = np.full(ngrid, np.nan)
        CorrSp = np.full(ngrid, np.nan)
        R2 = np.full(ngrid, np.nan)
        NSE = np.full(ngrid, np.nan)
        PBiaslow = np.full(ngrid, np.nan)
        PBiashigh = np.full(ngrid, np.nan)
        PBias = np.full(ngrid, np.nan)
        PBiasother = np.full(ngrid, np.nan)
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
                PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
                PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
                PBiasother[k] = np.sum(otherpred - othertarget) / np.sum(othertarget) * 100
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

    outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, CorrSp=CorrSp, R2=R2, NSE=NSE,
                   FLV=PBiaslow, FHV=PBiashigh, PBias=PBias, PBiasother=PBiasother, KGE=KGE, KGE12=KGE12, fdcRMSE=FDCRMSE,
                   lowRMSE=RMSElow, highRMSE=RMSEhigh, midRMSE=RMSEother)
    
    return outDict


def calculate_metrics_multi(args_list, model_outputs, y_obs_list, ensemble_type='max', out_dir=None):
    """
    Calculate stats for a multimodel ensemble.
    """
    stats_list = dict()

    for mod in args_list:
        args = args_list[mod]
        mod_out = model_outputs[mod]
        y_obs = y_obs_list[mod]

        if mod in ['SACSMA', 'marrmot_PRMS', 'farshid_hbv']:
            # Note for hydrodl HBV, calculations have already been done,
            # so skip this step.
            
            # Saving data            
            if out_dir:
                path = os.path.join(out_dir,"models\\671_sites_dp\\" + mod + "\\")
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

                # Test data (obs and model results).
                for key in mod_out[0].keys():
                    if len(mod_out[0][key].shape) == 3:
                        dim = 1
                    else:
                        dim = 0
                    concatenated_tensor = torch.cat([d[key] for d in mod_out], dim=dim)
                    file_name = key + ".npy"
                    np.save(os.path.join(path, file_name), concatenated_tensor.numpy())
                    # np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), concatenated_tensor.numpy())

                # Reading and flow observations.
                print(args['target'])
                for var in args["target"]:
                    item_obs = y_obs[:, :, args["target"].index(var)]
                    file_name = var + ".npy"
                    np.save(os.path.join(path, file_name), item_obs)
                    # np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), item_obs)


            ###################### calculations here ######################
            predLst = list()
            obsLst = list()
            flow_sim = torch.cat([d["flow_sim"] for d in mod_out], dim=1)
            flow_obs = y_obs[:, :, args["target"].index("00060_Mean")]
            predLst.append(flow_sim.numpy())
            obsLst.append(np.expand_dims(flow_obs, 2))

            # if args["temp_model_name"] != "None":
            #     temp_sim = torch.cat([d["temp_sim"] for d in mod_out], dim=1)
            #     temp_obs = y_obs[:, :, args["target"].index("00010_Mean")]
            #     predLst.append(temp_sim.numpy())
            #     obsLst.append(np.expand_dims(temp_obs, 2))

            # we need to swap axes here to have [basin, days], and remove redundant 
            # dimensions with np.squeeze().
            stats_list[mod] = [
                statError(np.swapaxes(x.squeeze(), 1, 0), np.swapaxes(y.squeeze(), 1, 0))
                for (x, y) in zip(predLst, obsLst)
            ]
        elif mod == 'hbv':
            stats_list[mod] = [statError(mod_out[:,:,0], y_obs.squeeze())]
        else:
            raise ValueError(f"Unsupported model type in `models`.")

    # Calculating final statistics for the whole set of basins.
    name_list = ["flow", "temp"]
    for st, name in zip(stats_list[mod], name_list):
        count = 0
        mdstd = np.zeros([len(st), 3])
        for key in st.keys():
            # Find the best result (e.g., the max, avg, median) and merge from each model.
            for i, mod in enumerate(args_list):
                if i == 0:
                    temp = stats_list[mod][0][key]
                    continue
                elif i == 1:
                    temp = np.stack((temp, stats_list[mod][0][key]), axis=1)
                else:
                    temp = np.hstack((temp, stats_list[mod][0][key].reshape(-1,1)))

            if len(args_list) > 1:
                if ensemble_type == 'max':
                    # print(temp, key)
                    temp = np.amax(temp, axis=1)
                    # print(temp, key)
                elif ensemble_type == 'avg':
                    temp = np.mean(temp, axis=1)
                elif ensemble_type == 'median':
                    temp = np.median(temp, axis=1)
                else:
                    raise ValueError("Invalid ensemble type specified.")
                
            median = np.nanmedian(temp)  # abs(i)
            std = np.nanstd(temp)  # abs(i)
            mean = np.nanmean(temp)  # abs(i)
            k = np.array([[median, std, mean]])
            mdstd[count] = k
            count = count + 1
            
        # mdstd displays the statistics for each error measure in stats_list.
        mdstd = pd.DataFrame(
            mdstd, index=st.keys(), columns=["median", "STD", "mean"]
        )
        # Save the data stats from the training run:
        if out_dir and len(args_list) > 1:
            path = os.path.join(out_dir, "multimodels\\671_sites_dp\\n_" + ensemble_type + "\\")
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                    
            mdstd.to_csv((os.path.join(path, "mdstd_" + name + ensemble_type +".csv")))
        elif out_dir:
            path = os.path.join(out_dir, "models\\671_sites_dp\\" + args_list[0] + "\\")
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            mdstd.to_csv((os.path.join(path, "mdstd_" + name +".csv")))
        else: continue

     # Show boxplots of the results
    plt.rcParams["font.size"] = 14
    keyLst = ["Bias", "RMSE", "ubRMSE", "NSE", "Corr"]
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        # for k in range(len(st)):
        data = st[statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
        dataBox.append(temp)
    labelname = [
        "Hybrid differentiable model"
    ]  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']

    xlabel = ["Bias ($\mathregular{deg}$C)", "RMSE", "ubRMSE", "NSE", "Corr"]
    fig = plot.plotBoxFig(
        dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8)
    )
    fig.patch.set_facecolor("white")
    boxPlotName = "PGML"
    fig.suptitle(boxPlotName, fontsize=12)
    plt.rcParams["font.size"] = 12
    # plt.savefig(
    #     os.path.join(args["out_dir"], args["testing_dir"], "Box_" + name + ".png")
    # )  # , dpi=500
    # fig.show()
    plt.close()

    torch.cuda.empty_cache()
    # print("Testing ended")

    return stats_list, mdstd



if __name__ == "__main__":
    # Fetch model predictions and observations.
    model_output, y_obs = test_multi_model()

    # Compute all statistics in `stats_list` and report summary stats across all
    # basins in mtstd.
    stats_list, mtstd = calculate_metrics_multi(args_list,
                                                model_outputs=model_output,
                                                y_obs_list=y_obs,
                                                ensemble_type=ENSEMBLE_TYPE,
                                                out_dir=OUT_DIR
                                                )
    
    print("Reporting median summary statistics: \n", mtstd['median'])

    # To avoid pytorch data leakage issues, clear g-ram cache.
    torch.cuda.empty_cache()
    print("Testing ended")