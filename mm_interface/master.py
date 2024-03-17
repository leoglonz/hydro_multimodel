"""
General scripts for running multimodel interface.

May decide to organize these later, but for now this file includes
everything not in functional.py, and not in existing files.
"""
import os
import torch
import platform


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
        device = torch.device("cpu")
    dtype = torch.float32

    return device, dtype



def set_platform_dirs():
    """
    Set output directory path to for systems with directory structures
    and locations.
    Currently supports: windows, mac os, and linux colab.

    outputs
        dir, str: output directory
    """
    if platform.system() == 'Windows':
        # Windows
        dir = os.path.join('D:','data','model_runs','hydro_multimodel_results')
    elif platform.system() == 'Darwin':
        # MacOs
        dir = os.path.join('Users','leoglonz','Desktop','water','data','model_runs','hydro_multimodel_results')
    elif platform.system() == 'Linux':
        # For Colab
        dir = os.path.join('content','drive','MyDrive','Colab','data','model_runs','hydro_multimodel_results'
    else:
        raise ValueError('Unsupported operating system.')
    
    return dir



def getModelDict(mlist, )
    """
    Create model and argument dictionaries to individual manage models in an
    ensemble interface.

    mlist is a list of models
    """
    models, arg_list = {}, {}
    for mod in mlist:
        if mod in ['dPLHBV_dyn','SACSMA_snow', 'marrmot_PRMS']
            models[str(mod)] = None
#################
    #  Continues here ##################      
##################

    return models, arg_list





def createDictFromKeys(keyList, mtd=0, dims=None, dat=None):
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
            d[k] = createTensor(dims)
        elif mtd == 11 or mtd == 'value':
            d[k] = createTensor(dims) + dat
        elif mtd == 2 or mtd == 'ref':
            d[k] = dat[..., kk]
        elif mtd == 21 or mtd == 'refClone':
            d[k] = dat[..., kk].clone()
    return d
