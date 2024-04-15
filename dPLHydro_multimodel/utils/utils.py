import os
import torch
import numpy as np
import random
import platform


def set_globals() -> None:
    """
    Select torch device and dtype global variables per user system.
    """ 
    global device, dtype

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    elif torch.backends.mps.is_available():
        # Use Mac M-series ARM architecture.
        device = torch.device('mps')
    else:
        device = torch.device("cpu")
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
    Fix the random seeds and behavior for reproducibility.
    Default seed is 0, None means random.
    """
    if seed == None:
        # Generate random seed
        randomseed = int(np.random.uniform(low=0, high=1e6))
        print("Random seed updated!")
    else:
        print("Setting seed 0.")
        random.seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)


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