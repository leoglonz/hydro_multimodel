"""
Functions here are used for training and testing differentiable model.
"""
import os
import numpy as np
import pandas as pd
import torch
import math
import time
from tqdm import tqdm
from post import stat, plot
import matplotlib.pyplot as plt
from core.data_processing.normalization import init_norm_stats, transNorm
from core.data_processing.data_loading import loadData
from core.data_processing.model import (
    No_iter_nt_ngrid,
    take_sample_train,
    take_sample_test,
    converting_flow_from_ft3_per_sec_to_mm_per_day
)



def train_differentiable_model(args, diff_model, lossFun, optim, start_epoch=1):
    if torch.cuda.is_available():
        diff_model = diff_model.to(args["device"])
        lossFun = lossFun.to(args["device"])
        torch.backends.cudnn.deterministic = True
        CUDA_LAUNCH_BLOCKING = 1

    # preparing training dataset
    dataset_dictionary = loadData(args, trange=args["t_train"])
    ### normalizing
    # creating the stats for normalization of NN inputs
    init_norm_stats(args, dataset_dictionary["x_NN"], dataset_dictionary["c_NN"], dataset_dictionary["obs"])
    # normalize
    x_NN_scaled = transNorm(args, dataset_dictionary["x_NN"], varLst=args["varT_NN"], toNorm=True)
    c_NN_scaled = transNorm(args, dataset_dictionary["c_NN"], varLst=args["varC_NN"], toNorm=True)
    c_NN_scaled = np.repeat(np.expand_dims(c_NN_scaled, 0), x_NN_scaled.shape[0], axis=0)
    del dataset_dictionary["x_NN"],   # no need the real values anymore
    dataset_dictionary["inputs_NN_scaled"] = np.concatenate((x_NN_scaled, c_NN_scaled), axis=2)
    del x_NN_scaled, c_NN_scaled   # we just need "inputs_NN_model" which is a combination of these two

    ngrid_train, nIterEp, nt, batchSize = No_iter_nt_ngrid("t_train", args, dataset_dictionary["inputs_NN_scaled"])
    diff_model.zero_grad()
    diff_model.train()

    # training
    for epoch in range(start_epoch, args["EPOCHS"] + 1):
        lossEp = 0
        t0 = time.time()
        print("Beginning training:")
        prog_str = "Epoch " + str(epoch) + "/" + str(args["EPOCHS"])

        for iIter in tqdm(range(1, nIterEp + 1),desc=prog_str,leave=False):
            dataset_dictionary_sample = take_sample_train(args, dataset_dictionary, ngrid_train, nt, batchSize)
            # Batch running of the differentiable model
            out_diff_model = diff_model(dataset_dictionary_sample)
            # loss function
            loss = lossFun(args, out_diff_model, dataset_dictionary_sample["obs_sample"])
            loss.backward()  # retain_graph=True
            optim.step()
            diff_model.zero_grad()
            lossEp = lossEp + loss.item()
            # if (iIter % 1 == 0) or (iIter == nIterEp):
            #     print(iIter, " from ", nIterEp, " in the ", epoch,
            #           "th epoch, and Loss is ", loss.item())

        lossEp = lossEp / nIterEp
        logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory".format(
            str(epoch), lossEp, time.time() - t0,
            int(torch.cuda.memory_allocated(device=args["device"]) * 0.001))
        print(logStr, end='\r', flush=True)

        if epoch % args["saveEpoch"] == 0:
            # save model
            modelFile = os.path.join(args["out_dir"], "model_Ep" + str(epoch) + ".pt")
            torch.save(diff_model, modelFile)
        if epoch == args["EPOCHS"]:
            print("*Last epoch*")
    
    print("- Training complete -")


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
    y_obs = converting_flow_from_ft3_per_sec_to_mm_per_day(args, dataset_dictionary["c_NN"],
                                                           dataset_dictionary["obs"][warm_up:, :, :])

    save_outputs(args, list_out_diff_model, y_obs, calculate_metrics=True)
    torch.cuda.empty_cache()
    print("Testing ended")


def save_outputs(args, list_out_diff_model, y_obs, calculate_metrics=True):
    for key in list_out_diff_model[0].keys():
        if len(list_out_diff_model[0][key].shape) == 3:
            dim = 1
        else:
            dim = 0
        concatenated_tensor = torch.cat([d[key] for d in list_out_diff_model], dim=dim)
        file_name = key + ".npy"
        np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), concatenated_tensor.numpy())

    # Reading flow observation
    for var in args["target"]:
        item_obs = y_obs[:, :, args["target"].index(var)]
        file_name = var + ".npy"
        np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), item_obs)

    if calculate_metrics == True:
        predLst = list()
        obsLst = list()
        flow_sim = torch.cat([d["flow_sim"] for d in list_out_diff_model], dim=1)
        flow_obs = y_obs[:, :, args["target"].index("00060_Mean")]
        predLst.append(flow_sim.numpy())
        obsLst.append(np.expand_dims(flow_obs, 2))
        if args["temp_model_name"] != "None":
            temp_sim = torch.cat([d["temp_sim"] for d in list_out_diff_model], dim=1)
            temp_obs = y_obs[:, :, args["target"].index("00010_Mean")]
            predLst.append(temp_sim.numpy())
            obsLst.append(np.expand_dims(temp_obs, 2))
        # we need to swap axes here to have [basin, days]
        statDictLst = [
            stat.statError(np.swapaxes(x.squeeze(), 1, 0), np.swapaxes(y.squeeze(), 1, 0))
            for (x, y) in zip(predLst, obsLst)
        ]
        ### save this file
        # median and STD calculation
        name_list = ["flow", "temp"]
        for st, name in zip(statDictLst, name_list):
            count = 0
            mdstd = np.zeros([len(st), 3])
            for key in st.keys():
                median = np.nanmedian(st[key])  # abs(i)
                STD = np.nanstd(st[key])  # abs(i)
                mean = np.nanmean(st[key])  # abs(i)
                k = np.array([[median, STD, mean]])
                mdstd[count] = k
                count = count + 1
            mdstd = pd.DataFrame(
                mdstd, index=st.keys(), columns=["median", "STD", "mean"]
            )
            mdstd.to_csv((os.path.join(args["out_dir"], args["testing_dir"], "mdstd_" + name + ".csv")))

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
            plt.savefig(
                os.path.join(args["out_dir"], args["testing_dir"], "Box_" + name + ".png")
            )  # , dpi=500
            # fig.show()
            plt.close()










