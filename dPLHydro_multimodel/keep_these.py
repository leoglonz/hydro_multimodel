"""
General scripts for running multimodel interface.

May decide to organize these later, but for now this file includes
everything not in functional.py, and not in existing files.
"""
import json
from multiprocessing.spawn import prepare
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import numpy as np
import torch
from data.load_data.normalizing import transNorm
from data.load_data.time import tRange2Array
from utils.stat import statError



# utils > master

def save_outputs(config, preds_list, y_obs) -> None:
    """
    Save outputs from a model.
    """
    for key in preds_list[0].keys():
        if len(preds_list[0][key].shape) == 3:
            # May need to flip 1 and 0 to save multimodels.
            dim = 1
        else:
            dim = 0

        concatenated_tensor = torch.cat([d[key] for d in preds_list], dim=dim)
        file_name = key + ".npy"        

        np.save(os.path.join(config['testing_dir'], file_name), concatenated_tensor.numpy())

    # Reading flow observation
    for var in config['target']:
        item_obs = y_obs[:, :, config['target'].index(var)]
        file_name = var + '.npy'
        np.save(os.path.join(config['testing_dir'], file_name), item_obs)















# utils > utils.py

def set_globals():
    """
    Select torch device and dtype global vars per user system.
    """ 
    global device, dtype

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    elif torch.backends.mps.is_available():
        # Use Mac M-series ARM architecture.
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    dtype = torch.float32

    return device, dtype


def set_platform_dir(path=None) -> str:
    """
    Set output directory path to for systems with directory structures
    and locations.
    Currently supports: windows, mac os, and linux colab.

    outputs: directory where model results will be stored.
    """
    if path != (None or ""):
        # if save path is already given in config, do nothing.
        return path
    elif platform.system() == 'Windows':
        # Windows
        return os.path.join('D:\\','code_repos','water','data','model_runs','hydro_multimodel_results')
    elif platform.system() == 'Darwin':
        # MacOs
        return os.path.join('Users','leoglonz','Desktop','water','data','model_runs','hydro_multimodel_results')
    elif platform.system() == 'Linux':
        # For Colab
        return os.path.join('content','drive','MyDrive','Colab','data','model_runs','hydro_multimodel_results')
    else:
        raise ValueError('Unsupported operating system.')
    

def randomseed_config(seed=0) -> None:
    """
    Fix the random seeds for reproducibility.
    seed = None -> random.
    """
    if seed == None:
        randomseed = int(np.random.uniform(low=0, high=1e6))
        pass

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass



def print_args(args):
    print("\033[1m" + "Basic Config Info" + "\033[0m")
    print(f'  {"Experiment Mode:":<20}{args.mode:<20}')
    print(f'  {"Ensemble Mode:":<20}{args.ensemble_type:<20}')

    for i, mod in enumerate(args.hydro_models):
        print(f'  {f"Model {i+1}:":<20}{mod:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.forcings:<20}')
    print(f'  {"Data Source:":<20}{Path(args.forcings).name:<20}')
    # print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Train Epochs:":<20}{args.epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Dropout:":<20}{args.dropout:<20}{"Hidden Size:":<20}{args.hidden_size:<20}')
    print(f'  {"Warmup:":<20}{args.warm_up:<20}{"Number of Models:":<20}{args.nmul:<20}')
    print(f'  {"Optimizer:":<20}{args.loss_function:<20}')
    print()

    print("\033[1m" + "Weighting Network Parameters" + "\033[0m")
    print(f'  {"Dropout:":<20}{args.weighting_nn.dropout:<20}{"Hidden Size:":<20}{args.weighting_nn.hidden_size:<20}')
    print(f'  {"Optimizer:":<20}{args.weighting_nn.loss_function:<20}{"Loss Factor:":<20}{args.weighting_nn.loss_factor:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.device:<20}{"GPU:":<20}{args.gpu_id:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    print()





# data > load data> data prep

def No_iter_nt_ngrid(time_range, args, x):
    nt, ngrid, nx = x.shape
    t = tRange2Array(time_range)
    if t.shape[0] < args['rho']:
        rho = t.shape[0]
    else:
        rho = args['rho']
    nIterEp = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - args['batch_size'] * rho / ngrid / (nt - args['warm_up']))
        )
    )
    return ngrid, nIterEp, nt, args['batch_size']


def selectSubset(args, x, iGrid, iT, rho, *, c=None, tupleOut=False, has_grad=False, warm_up=0):
    nx = x.shape[-1]
    nt = x.shape[0]
    # if x.shape[0] == len(iGrid):   #hack
    #     iGrid = np.arange(0,len(iGrid))  # hack
    #     if nt <= rho:
    #         iT.fill(0)

    if iT is not None:
        batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho + warm_up, batchSize, nx], requires_grad=has_grad)
        for k in range(batchSize):
            temp = x[np.arange(iT[k] - warm_up, iT[k] + rho), iGrid[k] : iGrid[k] + 1, :]
            xTensor[:, k : k + 1, :] = torch.from_numpy(temp)
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel
            # x = Ngrid * Ntime
            xTensor = torch.from_numpy(x[iGrid, :]).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(x[:, iGrid, :]).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho + warm_up, axis=1)
        cTensor = torch.from_numpy(temp).float()

        if tupleOut:
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        # out = out.cuda()
        out = out.to(args['device'])
    return out


def randomIndex(ngrid, nt, dimSubset, warm_up=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+warm_up, nt - rho, [batchSize])
    return iGrid, iT


def take_sample_train(args, dataset_dictionary, ngrid_train, nt, batchSize):
    dimSubset = [batchSize, args['rho']]
    iGrid, iT = randomIndex(ngrid_train, nt, dimSubset, warm_up=args['warm_up'])
    dataset_dictionary_sample = dict()
    dataset_dictionary_sample['iGrid'] = iGrid
    dataset_dictionary_sample['inputs_nn_scaled'] = selectSubset(args, dataset_dictionary['inputs_nn_scaled'],
                                                                        iGrid, iT, args['rho'], has_grad=False,
                                                                        warm_up=args['warm_up'])
    dataset_dictionary_sample['c_nn'] = torch.tensor(
        dataset_dictionary['c_nn'][iGrid], device=args['device'], dtype=torch.float32
    )
    # collecting observation samples
    dataset_dictionary_sample['obs'] = selectSubset(
        args, dataset_dictionary['obs'], iGrid, iT, args['rho'], has_grad=False, warm_up=args['warm_up']
    )[args['warm_up']:, :, :]
    # dataset_dictionary_sample['obs'] = converting_flow_from_ft3_per_sec_to_mm_per_day(args,
    #                                                                                          dataset_dictionary_sample[
    #                                                                                              'c_nn'],
    #                                                                                          obs_sample_v)
    # Hydro model sampling
    
    dataset_dictionary_sample['x_hydro_model'] = selectSubset(
        args, dataset_dictionary['x_hydro_model'], iGrid, iT, args['rho'], has_grad=False, warm_up=args['warm_up']
    )
    dataset_dictionary_sample['c_hydro_model'] = torch.tensor(
        dataset_dictionary['c_hydro_model'][iGrid], device=args['device'], dtype=torch.float32
    )

    return dataset_dictionary_sample


def take_sample_test(args, dataset_dictionary, iS, iE):
    dataset_dictionary_sample = dict()
    for key in dataset_dictionary.keys():
        if len(dataset_dictionary[key].shape) == 3:
            # we need to remove the warm up period for all except airT_memory and inputs for temp model
            if (key == 'airT_mem_temp_model') or (key == 'x_temp_model') or (key == 'x_hydro_model') or (
                    key == 'inputs_nn_scaled'):
                warm_up = 0
            else:
                warm_up = args['warm_up']
            dataset_dictionary_sample[key] = dataset_dictionary[key][warm_up:, iS: iE, :].to(
                args['device'])
        elif len(dataset_dictionary[key].shape) == 2:
            dataset_dictionary_sample[key] = dataset_dictionary[key][iS: iE, :].to(
                args['device'])
    return dataset_dictionary_sample


def breakdown_params(self, params_all):
        params_dict = dict()
        params_hydro_model = params_all[:, :, :self.ny]

        # hydro params
        params_dict['hydro_params_raw'] = torch.sigmoid(
            params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.config['nmul']]).view(
            params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
            self.config['nmul'])
        # routing params
        if self.config['routing_hydro_model'] == True:
            params_dict['conv_params_hydro'] = torch.sigmoid(
                params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.config['nmul']:])
        else:
            params_dict['conv_params_hydro'] = None
        return params_dict