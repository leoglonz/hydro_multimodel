import importlib

import numpy as np


def get_loss_func(args, obs):
    # args['target'] = ['00060_Mean']
    # module = importlib.import_module(args['loss_function'])
    spec = importlib.util.spec_from_file_location("RmseLoss_flow_comb", "/data/lgl5139/hydro_multimodel/dPLHydro_multimodel/models/loss_functions/RmseLoss_flow_comb.py") #args['loss_function'], './models/loss_functions/' + args['loss_function'] + '.py')
    module = spec.loader.load_module()
    loss_function_default = getattr(module, args['loss_function'])
    
    if args['loss_function'] in ['RmseLoss_flow_temp', 'RmseLoss_flow_temp_BFI', 'RmseLoss_flow_temp_BFI_PET', 'RmseLoss_BFI_temp']:
        lossFun = loss_function_default(w1=args['loss_function_weights']['w1'],
                                        w2=args['loss_function_weights']['w2'])
    
    elif args['loss_function'] == 'NSEsqrtLoss_flow_temp':
        std_obs_flow = np.nanstd(obs[:, :, args['target'].index('00060_Mean')], axis=0)
        std_obs_flow[std_obs_flow != std_obs_flow] = 1.0

        std_obs_temp = np.nanstd(obs[:, :, args['target'].index('00010_Mean')], axis=0)
        std_obs_temp[std_obs_temp != std_obs_temp] = 1.0

        lossFun = loss_function_default(stdarray_flow=std_obs_flow,
                                        stdarray_temp=std_obs_temp)
    
    elif args['loss_function'] == 'NSEsqrtLoss_flow':
        std_obs_flow = np.nanstd(obs[:, :, args['target'].index('00060_Mean')], axis=0)
        std_obs_flow[std_obs_flow != std_obs_flow] = 1.0
        lossFun = loss_function_default(stdarray_flow=std_obs_flow)
    else:
        lossFun = loss_function_default()
    return lossFun
