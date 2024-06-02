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

# Set list of supported hydro models here:
supported_models = ['HBV', 'dPLHBV_stat', 'dPLHBV_dyn', 'SACSMA', 'SACSMA_snow',
                   'marrmot_PRMS']


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



def set_platform_dir():
    """
    Set output directory path to for systems with directory structures
    and locations.
    Currently supports: windows, mac os, and linux colab.

    outputs
        dir: output directory
    """
    if platform.system() == 'Windows':
        # Windows
        dir = os.path.join('D:\\','code_repos','water','data','model_runs','hydro_multimodel_results')
    elif platform.system() == 'Darwin':
        # MacOs
        dir = os.path.join('Users','leoglonz','Desktop','water','data','model_runs','hydro_multimodel_results')
    elif platform.system() == 'Linux':
        # For Colab
        dir = os.path.join('content','drive','MyDrive','Colab','data','model_runs','hydro_multimodel_results')
    else:
        raise ValueError('Unsupported operating system.')
    
    return dir


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


def create_tensor(dims, requires_grad=False):
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
        if mtd == 0 or mtd is None or mtd == 'None':
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


def save_outputs(config, preds_list, y_obs):
    for key in preds_list[0].keys():
        if len(preds_list[0][key].shape) == 3:
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


def create_output_dirs(config):
    out_folder = config['nn_model'] + \
             '_E' + str(config['epochs']) + \
             '_R' + str(config['rho'])  + \
             '_B' + str(config['batch_size']) + \
             '_H' + str(config['hidden_size']) + \
             '_n' + str(config['nmul']) + \
             '_' + str(config['random_seed'])

    # make a folder for static and dynamic parametrization
    if config['dyn_hydro_params']['HBV'] != []:
        dyn_state = 'dynamic_para'
    else:
        dyn_state = 'static_para'
    if config['freeze_para_nn'] == True:
        para_state = 'frozen_pnn'
    else:
        para_state = 'free_pnn'

    test_dir = 'test' + str(config['test']['start_time'][:4]) + '_' + str(config['test']['end_time'][:4])

    test_path = os.path.join(config['output_dir'], para_state, out_folder, dyn_state, test_dir)

    config['testing_dir'] = test_path
    os.makedirs(test_path, exist_ok=True)
    
    config['output_dir'] = os.path.join(config['output_dir'], para_state, out_folder, dyn_state)

    # saving the config file in output directory
    config_file = json.dumps(config)
    config_path = os.path.join(config['output_dir'], 'config_file.json')
    if os.path.exists(config_path):
        os.remove(config_path)
    f = open(config_path, 'w')
    f.write(config_file)
    f.close()

    return config


# def loadModel(out, epoch=None):
#     if epoch is None:
#         mDict = readMasterFile(out)
#         epoch = mDict['train']['nEpoch']
#     model = hydroDL.model.train.loadModel(out, epoch)
#     return model


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model
