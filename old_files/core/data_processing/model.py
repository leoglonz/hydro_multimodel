"""
Functions related specifically to DL model runs.
May contain some near-identical functions in the mean time while merging models.
"""
import numpy as np
import torch
from core.data_processing.data_prep import randomIndex, selectSubset
from core.utils.time import tRange2Array


def train_val_test_split_action1(set_name, args, time1, x_total, y_total):
    t = tRange2Array(args[set_name])
    c, ind1, ind2 = np.intersect1d(time1, t, return_indices=True)
    x = x_total[:, ind1, :]
    y = y_total[:, ind1, :]
    ngrid, nt, nx = x.shape
    if t.shape[0] < args["rho"]:
        rho = t.shape[0]
    else:
        rho = args["rho"]

    return x, y, ngrid, nt, args["batch_size"]


def take_sample_train(args, dataset_dictionary, ngrid_train, nt, batchSize):
    dimSubset = [batchSize, args["rho"]]
    iGrid, iT = randomIndex(ngrid_train, nt, dimSubset, warm_up=args["warm_up"])
    dataset_dictionary_sample = dict()
    dataset_dictionary_sample["inputs_NN_scaled_sample"] = selectSubset(args, dataset_dictionary["inputs_NN_scaled"],
                                                                        iGrid, iT, args["rho"], has_grad=False,
                                                                        warm_up=args["warm_up"])
    dataset_dictionary_sample["c_NN_sample"] = torch.tensor(
        dataset_dictionary["c_NN"][iGrid], device=args["device"], dtype=torch.float32
    )
    # collecting observation samples
    obs_sample_v = selectSubset(
        args, dataset_dictionary["obs"], iGrid, iT, args["rho"], has_grad=False, warm_up=args["warm_up"]
    )[args["warm_up"]:, :, :]
    dataset_dictionary_sample["obs_sample"] = converting_flow_from_ft3_per_sec_to_mm_per_day(args,
                                                                                             dataset_dictionary_sample[
                                                                                                 "c_NN_sample"],
                                                                                             obs_sample_v)
    # Hydro model sampling
    if args["hydro_model_name"] != "None":
        dataset_dictionary_sample["x_hydro_model_sample"] = selectSubset(
            args, dataset_dictionary["x_hydro_model"], iGrid, iT, args["rho"], has_grad=False, warm_up=args["warm_up"]
        )
        dataset_dictionary_sample["c_hydro_model_sample"] = torch.tensor(
            dataset_dictionary["c_hydro_model"][iGrid], device=args["device"], dtype=torch.float32
        )
    # temperture model sampling
    if args["temp_model_name"] != "None":
        dataset_dictionary_sample["x_temp_model_sample"] = selectSubset(
            args, dataset_dictionary["x_temp_model"], iGrid, iT, args["rho"], has_grad=False, warm_up=args["warm_up"]
        )  # [warm_up:,:, :]there is no need for warm up in temp section yet
        dataset_dictionary_sample["c_temp_model_sample"] = torch.tensor(
            dataset_dictionary["c_temp_model"][iGrid], device=args["device"], dtype=torch.float32
        )

    return dataset_dictionary_sample


def take_sample_test(args, dataset_dictionary, iS, iE):
    dataset_dictionary_sample = dict()
    for key in dataset_dictionary.keys():
        if len(dataset_dictionary[key].shape) == 3:
            dataset_dictionary_sample[key + "_sample"] = dataset_dictionary[key][:, iS: iE, :].to(
                args["device"])
        elif len(dataset_dictionary[key].shape) == 2:
            dataset_dictionary_sample[key + "_sample"] = dataset_dictionary[key][iS: iE, :].to(
                args["device"])
            
    return dataset_dictionary_sample


def converting_flow_from_ft3_per_sec_to_mm_per_day(args, c_NN_sample, obs_sample):
    varTar_NN = args["target"]
    obs_flow_v = obs_sample[:, :, varTar_NN.index("00060_Mean")]
    varC_NN = args["varC_NN"]
    if "DRAIN_SQKM" in varC_NN:
        area_name = "DRAIN_SQKM"
    elif "area_gages2" in varC_NN:
        area_name = "area_gages2"
    area = (c_NN_sample[:, varC_NN.index(area_name)]).unsqueeze(0).repeat(obs_flow_v.shape[0], 1)
    obs_sample[:, :, varTar_NN.index("00060_Mean")] = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6)) # convert ft3/s to mm/day

    return obs_sample


def No_iter_nt_ngrid(set_name, args, x):
    ngrid, nt, nx = x.shape
    t = tRange2Array(args[set_name])
    if t.shape[0] < args["rho"]:
        rho = t.shape[0]
    else:
        rho = args["rho"]
    nIterEp = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - args["batch_size"] * rho / ngrid / nt)
        )
    )
    
    return ngrid, nIterEp, nt, args["batch_size"]
