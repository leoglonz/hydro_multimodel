import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import json
import random
import numpy as np
import pandas as pd
import polars as pl
import torch
import xarray as xr
import zarr
import os
from conf.config import Config
from tqdm import tqdm

log = logging.getLogger(__name__)



def set_system_spec(cuda_device: Optional[int] = None) -> Tuple[torch.device, torch.dtype]:
    """
    Sets appropriate torch device and dtype for current system.
    
    Args:
        user_selected_cuda (Optional[int]): The user-specified CUDA device index. Defaults to None.
    
    Returns:
        Tuple[torch.device, torch.dtype]: A tuple containing the device and dtype.
    """
    if cuda_device != None:
        if torch.cuda.is_available() and cuda_device < torch.cuda.device_count():
            device = torch.device(f'cuda:{cuda_device}')
            torch.cuda.set_device(device)   # Set as active device.
        else:
            raise ValueError(f"Selected CUDA device {cuda_device} is not available.")
        
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        torch.cuda.set_device(device)   # Set as active device.

    elif torch.backends.mps.is_available():
        # Use Mac M-series ARM architecture.
        device = torch.device('mps')

    else:
        device = torch.device('cpu')
    
    dtype = torch.float32
    return device, dtype


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
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.use_deterministic_algorithms(True)
    except:
        pass
    

def create_output_dirs(config) -> dict:
    out_folder = config['nn_model'] + \
             '_E' + str(config['epochs']) + \
             '_R' + str(config['rho'])  + \
             '_B' + str(config['batch_size']) + \
             '_H' + str(config['hidden_size']) + \
             '_n' + str(config['nmul']) + \
             '_' + str(config['random_seed'])

    # Make a folder for static or dynamic parametrization
    if config['dyn_hydro_params']['HBV'] != []:
        # If one model has dynamic params, all of them should.
        dyn_state = 'dynamic_para'
    else:
        dyn_state = 'static_para'
    if config['ensemble_type'] == 'None':
        para_state = 'no_ensemble'
    elif config['freeze_para_nn'] == True:
        para_state = 'frozen_pnn'
    else:
        para_state = 'free_pnn'

    config['output_dir'] = os.path.join(config['output_dir'], para_state, out_folder, dyn_state)

    test_dir = 'test' + str(config['test']['start_time'][:4]) + '_' + str(config['test']['end_time'][:4])
    test_path = os.path.join(config['output_dir'], test_dir)
    config['testing_dir'] = test_path
    os.makedirs(test_path, exist_ok=True)
    
    # saving the config file in output directory
    config_file = json.dumps(config)
    config_path = os.path.join(config['output_dir'], 'config_file.json')
    if os.path.exists(config_path):
        os.remove(config_path)
    with open(config_path, 'w') as f:
        f.write(config_file)

    return config


def save_model(config, model, model_name, epoch, create_dirs=False) -> None:
    """
    Save ensemble or single models.
    """
    # If the model folder has not been created, do it here.
    if create_dirs: create_output_dirs(config)

    save_dir = str(model_name) + '_model_Ep' + str(epoch) + '.pt'
    os.makedirs(save_dir, exist_ok=True)

    full_path = os.path.join(config['output_dir'], save_dir)
    torch.save(model, full_path)


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


def show_args(config) -> None:
    """
    From Jiangtao Liu.
    Use to display critical configuration settings in a clean format.
    """
    print("\033[1m" + "Basic Config Info" + "\033[0m")
    print(f'  {"Experiment Mode:":<20}{config.mode:<20}')
    print(f'  {"Ensemble Mode:":<20}{config.ensemble_type:<20}')

    for i, mod in enumerate(config.hydro_models):
        print(f'  {f"Model {i+1}:":<20}{mod:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{config.forcings:<20}')
    print(f'  {"Data Source:":<20}{Path(config.forcings).name:<20}')
    # print(f'  {"Checkpoints:":<20}{config.checkpoints:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Train Epochs:":<20}{config.epochs:<20}{"Batch Size:":<20}{config.batch_size:<20}')
    print(f'  {"Dropout:":<20}{config.dropout:<20}{"Hidden Size:":<20}{config.hidden_size:<20}')
    print(f'  {"Warmup:":<20}{config.warm_up:<20}{"Number of Models:":<20}{config.nmul:<20}')
    print(f'  {"Optimizer:":<20}{config.loss_function:<20}')
    print()

    print("\033[1m" + "Weighting Network Parameters" + "\033[0m")
    print(f'  {"Dropout:":<20}{config.weighting_nn.dropout:<20}{"Hidden Size:":<20}{config.weighting_nn.hidden_size:<20}')
    print(f'  {"Optimizer:":<20}{config.weighting_nn.loss_function:<20}{"Loss Factor:":<20}{config.weighting_nn.loss_factor:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{config.device:<20}{"GPU:":<20}{config.gpu_id:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    print()
    