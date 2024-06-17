import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
import xarray as xr
import zarr
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
    """
    Create a new directory for model files.

    Modified from dPL_Hydro_SNTEMP @ Farshid Rahmani.
    """
    out_folder = config['pnn_model'] + \
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
    if config['ensemble_type'] == 'none':
        para_state = 'no_ensemble'
    elif config['ensemble_type'] == 'frozen_pnn':
        para_state = 'frozen_pnn'
    elif config['ensemble_type'] == 'free_pnn':
        para_state = 'free_pnn'
    else:
        raise ValueError("Unsupported ensemble type specified.")
    ensemble_name = ""
    for mod in config['hydro_models']:
        ensemble_name += mod + "_"

    config['output_dir'] = os.path.join(config['output_dir'], para_state, out_folder, dyn_state, ensemble_name)

    test_dir = 'test' + str(config['test']['start_time'][:4]) + '_' + str(config['test']['end_time'][:4])
    test_path = os.path.join(config['output_dir'], test_dir)
    config['testing_dir'] = test_path

    if (config['mode'] == 'test') and (os.path.exists(config['output_dir']) == False):
        raise FileNotFoundError(f"Model directory {config['output_dir']} was not found. Check configurations.")
    print(config['output_dir'])
    os.makedirs(test_path, exist_ok=True)
    
    # Saving the config file in output directory.
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

    save_name = str(model_name) + '_model_Ep' + str(epoch) + '.pt'
    # os.makedirs(save_name, exist_ok=True)

    full_path = os.path.join(config['output_dir'], save_name)
    torch.save(model, full_path)


def save_outputs(config, preds_list, y_obs, create_dirs=False) -> None:
    """
    Save outputs from a model.
    """
    if create_dirs: create_output_dirs(config)

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


def load_model(config, model_name, epoch):
    """
    Load trained pytorch models.
    
    Args:
        config (dict): Configuration dictionary with paths and model settings.
        model_name (str): Name of the model to load.
        epoch (int): Epoch number to load the specific state of the model.
        
    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    model_name = str(model_name) + '_model_Ep' + str(epoch) + '.pt'
    # model_path = os.path.join(config['output_dir'], model_name)
    # try:
    #     self.model_dict[model] = torch.load(model_path).to(self.config['device']) 
    # except:
    #     raise FileNotFoundError(f"Model file {model_path} was not found. Check that epochs and hydro models in your config are correct.")

    # # Construct the path where the model is saved
    # model_file_name = f"{model_name}_epoch_{epoch}.pth"
    # model_path = os.path.join(config['model_dir'], model_file_name)
    
    # # Ensure the model file exists
    # if not os.path.isfile(model_path):
    #     raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    # return torch.load(model_path)
    
    # Retrieve the model class from config (assuming it's stored in the config)
    # model_class = config['model_classes'][model_name]
    
    # Initialize the model (assumes model classes are callable and take no arguments)
    # model = model_class()
    # Load the state_dict into the model
    # model.load_state_dict(state_dict)
    
    # return model


def show_args(config) -> None:
    """
    From Jiangtao Liu.
    Use to display critical configuration settings in a clean format.
    """
    print()
    print("\033[1m" + "Current Configuration" + "\033[0m")
    print(f'  {"Experiment Mode:":<20}{config.mode:<20}')
    print(f'  {"Ensemble Mode:":<20}{config.ensemble_type:<20}')

    for i, mod in enumerate(config.hydro_models):
        print(f'  {f"Model {i+1}:":<20}{mod:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data Source:":<20}{config.observations.name:<20}')
    if config.mode != 'test':
        print(f'  {"Train Range :":<20}{config.train.start_time:<20}{config.train.end_time:<20}')
    if config.mode != 'train':
        print(f'  {"Test Range :":<20}{config.test.start_time:<20}{config.test.end_time:<20}')
    if config.use_checkpoint == True:
        print(f'  {"Resuming training from epoch:":<20}{config.checkpoint.start_epoch:<20}')
    print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Train Epochs:":<20}{config.epochs:<20}{"Batch Size:":<20}{config.batch_size:<20}')
    print(f'  {"Dropout:":<20}{config.dropout:<20}{"Hidden Size:":<20}{config.hidden_size:<20}')
    print(f'  {"Warmup:":<20}{config.warm_up:<20}{"Concurrent Models:":<20}{config.nmul:<20}')
    print(f'  {"Optimizer:":<20}{config.loss_function:<20}')
    print()

    print("\033[1m" + "Weighting Network Parameters" + "\033[0m")
    print(f'  {"Dropout:":<20}{config.weighting_nn.dropout:<20}{"Hidden Size:":<20}{config.weighting_nn.hidden_size:<20}')
    print(f'  {"Method:":<20}{config.weighting_nn.method:<20}{"Loss Factor:":<20}{config.weighting_nn.loss_factor:<20}')
    print(f'  {"Loss Lower Bound:":<20}{config.weighting_nn.loss_lower_bound:<20}{"Loss Upper Bound:":<20}{config.weighting_nn.loss_upper_bound:<20}')
    print(f'  {"Optimizer:":<20}{config.weighting_nn.loss_function:<20}')
    print()

    print("\033[1m" + "Machine" + "\033[0m")
    print(f'  {"Use Device:":<20}{str(config.device):<20}')
    print()
    