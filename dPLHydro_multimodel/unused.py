"""
General scripts for running multimodel interface.

May decide to organize these later, but for now this file includes
everything not in functional.py, and not in existing files.
"""
import json
import os
import platform

import numpy as np
import torch
from data.load_data.time import tRange2Array
from utils.stat import statError
from core.load_data.normalizing import transNorm

import numpy as np


# utils > master

# Set list of supported hydro models here:
supported_models = ['HBV', 'dPLHBV_stat', 'dPLHBV_dyn', 'SACSMA', 'SACSMA_snow',
                   'marrmot_PRMS']

def get_model_dict(modList):
    """
    Create model and argument dictionaries to individual manage models in an
    ensemble interface.
    
    Inputs:
        modList: list of models.
    """
    models, arg_list = {}, {}
    for mod in modList:
        if mod in supported_models:
            models[mod] = None
            arg_list[mod] = config[mod]
        else:
            raise ValueError(f"Unsupported model type", mod)
    return models, arg_list


def make_tensor(*values, has_grad=False, dtype=torch.float32, device="cuda"):

    if len(values) > 1:
        tensor_list = []
        for value in values:
            t = torch.tensor(value, requires_grad=has_grad, dtype=dtype, device=device)
            tensor_list.append(t)
    else:
        for value in values:
            if type(value) != torch.Tensor:
                tensor_list = torch.tensor(
                    value, requires_grad=has_grad, dtype=dtype, device=device
                )
            else:
                tensor_list = value.clone().detach()
    return tensor_list



def create_output_dirs(args):
    seed = args["randomseed"][0]
    # checking rho value first
    t = tRange2Array(args["t_train"])
    if t.shape[0] < args["rho"]:
        args["rho"] = t.shape[0]

    # checking the directory
    if not os.path.exists(args["output_model"]):
        os.makedirs(args["output_model"])
    if args["hydro_model_name"]!= "None":
        hydro_name = "_" + args["hydro_model_name"]
    else:
        hydro_name = ""
    if args["temp_model_name"]!= "None":
        temp_name = "_" + args["temp_model_name"]
    else:
        temp_name = ""

    out_folder = args["NN_model_name"] + \
            hydro_name + \
            temp_name + \
            '_E' + str(args['EPOCHS']) + \
             '_R' + str(args['rho']) + \
             '_B' + str(args['batch_size']) + \
             '_H' + str(args['hidden_size']) + \
             "_tr" + str(args["t_train"][0])[:4] + "_" + str(args["t_train"][1])[:4] + \
            "_n" + str(args["nmul"]) + \
            "_" + str(seed)

    if not os.path.exists(os.path.join(args["output_model"], out_folder)):
        os.makedirs(os.path.join(args["output_model"], out_folder))

    ## make a folder for static and dynamic parametrization
    dyn_params = ""
    if args["hydro_model_name"]!= "None":
        if len(args["dyn_params_list_hydro"]) > 0:
            dyn_list_sorted = sorted(args["dyn_params_list_hydro"])
            for i in dyn_list_sorted:
                dyn_params = dyn_params + i + "_"
        else:
            dyn_params = "hydro_stat_"
    if args["temp_model_name"]!= "None":
        if len(args["dyn_params_list_temp"]) > 0:
            dyn_list_sorted = sorted(args["dyn_params_list_temp"])
            for i in dyn_list_sorted:
                dyn_params = dyn_params + i + "_"
        else:
            dyn_params = dyn_params + "temp_stat"

    testing_dir = "ts" + str(args["t_test"][0])[:4] + "_" + str(args["t_test"][1])[:4]
    if not os.path.exists(os.path.join(args["output_model"], out_folder, dyn_params, testing_dir)):
        os.makedirs(os.path.join(args["output_model"], out_folder, dyn_params, testing_dir))
    # else:
    #     shutil.rmtree(os.path.join(args['output']['model'], out_folder))
    #     os.makedirs(os.path.join(args['output']['model'], out_folder))
    args["out_dir"] = os.path.join(args["output_model"], out_folder, dyn_params)
    args["testing_dir"] = testing_dir

    # saving the args file in output directory
    config_file = json.dumps(args)
    config_path = os.path.join(args["out_dir"], "config_file.json")
    if os.path.exists(config_path):
        os.remove(config_path)
    f = open(config_path, "w")
    f.write(config_file)
    f.close()

    return args


def update_args(args, **kw):
    for key in kw:
        if key in args:
            try:
                args[key] = kw[key]
            except ValueError:
                print("Something went wrong in args when updating " + key)
        else:
            print("didn't find " + key + " in args")
    return args



def create_tensor(dims, requires_grad=False) -> torch.Tensor:
    """
    A small function to centrally manage device, data types, etc., of new arrays.
    """
    return torch.zeros(dims,requires_grad=requires_grad,dtype=dtype).to(device)


def create_dict_from_keys(keyList, mtd=0, dims=None, dat=None):
    """
    A modular dictionary initializer from C. Shen.

    mtd = 
        0: Init keys to None,
        1: Init keys to zero tensors,
        11: Init keys to tensors with the same vals as `dat`,
        2: Init keys to slices of `dat`,
        21: Init keys with cloned slices of `dat`.
    """
    d = {}
    for kk, k in enumerate(keyList):
        if mtd == 0 or mtd is None or mtd == "None":
            d[k] = None
        elif mtd == 1 or mtd == 'zeros':
            d[k] = create_tensor(dims)
        elif mtd == 11 or mtd == 'value':
            d[k] = create_tensor(dims) + dat
        elif mtd == 2 or mtd == 'ref':
            d[k] = dat[..., kk]
        elif mtd == 21 or mtd == 'refClone':
            d[k] = dat[..., kk].clone()
    return d


def save_output(args_list, preds, y_obs, out_dir) -> None:
    """
    Save extracted test preds and obs for all models.
    """
    for i, mod in enumerate(args_list):
        if i == 0:
            multim_dir = str(mod)
        else:
            multim_dir += '_' + str(mod)

        out_dir = os.path.join(out_dir, multim_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        arg = args_list[list(args_list)[1]]
        dir = 'multim_E' + str(arg['EPOCHS']) + '_B' + str(arg['batch_size']) + '_R' + str(arg['rho']) +  '_BT' + str(arg['warm_up']) + '_H' + str(arg['hidden_size']) + '_tr1980_1995_n' + str(arg['nmul'])

        np.save(os.path.join(out_dir, 'preds_' + dir + '.npy'), preds)
        np.save(os.path.join(out_dir, 'obs_' + dir + '.npy'), y_obs)



# core > load data > data_prep

def scaling(args, x, y, c):
    """
    creates our datasets
    :param set_name:
    :param args:
    :param time1:
    :param x_total_raw:
    :param y_total_raw:
    :return:  x, y, ngrid, nIterEp, nt
    """
    # initcamels(args, x, y)
    # Normalization
    x_total_scaled = transNorm(
        x, args['var_t_nn'] + args['var_c_nn'], toNorm=True
    )
    y_scaled = transNorm(y, args['target'], toNorm=True)
    c_scaled = transNorm(c, args['var_c_nn'], toNorm=True)
    return x_total_scaled, y_scaled, c_scaled


def train_val_test_split(set_name, args, time1, x_total, y_total):
    t = tRange2Array(args[set_name])
    c, ind1, ind2 = np.intersect1d(time1, t, return_indices=True)
    x = x_total[:, ind1, :]
    y = y_total[:, ind1, :]
    return x, y


def train_val_test_split_action1(set_name, args, time1, x_total, y_total):
    t = tRange2Array(args[set_name])
    c, ind1, ind2 = np.intersect1d(time1, t, return_indices=True)
    x = x_total[:, ind1, :]
    y = y_total[:, ind1, :]
    ngrid, nt, nx = x.shape
    if t.shape[0] < args['rho']:
        rho = t.shape[0]
    else:
        rho = args['rho']


    return x, y, ngrid, nt, args['batch_size']


# def load_df(args):
#     """
#     A function that loads the data into a
#     :return:
#     """
#     df, x, y, c, c_hydro_model, x_hydro_model, c_SNTEMP, x_SNTEMP = master.loadData(args)
#     nx = x.shape[-1] + c.shape[-1]
#     x_total = np.zeros((x.shape[0], x.shape[1], nx))
#     nx_SNTEMP = x_SNTEMP.shape[-1] + c_SNTEMP.shape[-1]
#     x_tot_SNTEMP = np.zeros((x.shape[0], x.shape[1], nx_SNTEMP))
#     ct = np.repeat(c, repeats=x.shape[1], axis=0)
#     for k in range(x.shape[0]):
#         x_total[k, :, :] = np.concatenate(
#             (x[k, :, :], np.tile(c[k], (x.shape[1], 1))), axis=1
#         )
#         x_tot_SNTEMP[k, :, :] = np.concatenate(
#             (x_SNTEMP[k, :, :], np.tile(c_SNTEMP[k], (x_SNTEMP.shape[1], 1))), axis=1
#         )
#
#
#     # streamflow values should not be negative
#     # vars = args['optData']['varT'] + args['optData']['varC']
#     # x_total[x_total[:, :, vars.index("00060_Mean")] < 0] = 0
#     return np.float32(x_total), np.float32(y), np.float32(c), np.float32(c_hydro_model), \
#         np.float32(x_hydro_model), np.float32(c_SNTEMP), np.float32(x_tot_SNTEMP)


def create_tensor_list(x, y):
    """
    we want to return the :
    x_list = [[[basin_1, num_samples_x, num_attr_x], [basin_1, num_samples_y, num_attr_y]]
        .
        .
        .
        [[basin_20, num_samples_x, num_attr_x], [basin_20, num_samples_y, num_attr_y]]]
    :param data:
    :return:
    """
    tensor_list = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            _var = (torch.tensor(x[i][j][:, :]), y[i, j])
            tensor_list.append(_var)
    return tensor_list


def create_tensor(rho, mini_batch, x, y):
    """
    Creates a data tensor of the input variables and incorporates a sliding window of rho
    :param mini_batch: min batch length
    :param rho: the seq len
    :param x: the x data
    :param y: the y data
    :return:
    """
    j = 0
    k = rho
    _sample_data_x = []
    _sample_data_y = []
    for i in range(x.shape[0]):
        _list_x = []
        _list_y = []
        while k < x[0].shape[0]:
            """In the format: [total basins, basin, days, attributes]"""
            _list_x.append(x[1, j:k, :])
            _list_y.append(y[1, j:k, 0])
            j += mini_batch
            k += mini_batch
        _sample_data_x.append(_list_x)
        _sample_data_y.append(_list_y)
        j = 0
        k = rho
    sample_data_x = torch.tensor(_sample_data_x).float()
    sample_data_y = torch.tensor(_sample_data_y).float()
    return sample_data_x, sample_data_y

