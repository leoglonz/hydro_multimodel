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
