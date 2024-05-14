# import logging

# import numpy as np
# import torch
# from conf.config import Config
# from data.load_data.data_prep import take_sample_test
# from data.load_data.dataFrame_loading import loadData
# from data.load_data.normalizing import transNorm
# from data.utils.Dates import Dates
# from models.multimodels.ensemble_network import EnsembleWeights
# from models.multimodels.multimodel_handler import MultimodelHandler
# from utils.master import save_output
# from utils.utils import set_globals

# log = logging.getLogger(__name__)


# class TestModel:
#     """
#     High-level multimodel testing handler; retrieves and formats testing data,
#     initializes all individual models, and runs testing.
#     """
#     def __init__(self, config: Config):
#         self.config = config
#         self.config['device'], self.config['dtype'] = set_globals()

#         # Initializing collection of differentiable hydrology models and their optimizers.
#         # Training this object will parallel train all hydro models specified for ensemble.
#         self.dplh_model_handler = MultimodelHandler(self.config).to(self.config['device'])

#         # Initialize the weighting LSTM.
#         self.ensemble_lstm = EnsembleWeights(self.config).to(self.config['device'])

#     def run(self):
#         log.info(f"Testing model: {self.config['name']}")

#         # Preparing training data.
#         # Formatting date ranges:
#         self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
#         self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
#         self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

#         # Read data for the test time range
#         dataset_dict = loadData(self.config, trange=self.test_trange)
        
#         warm_up = self.config['warm_up']
#         nmul = self.config['nmul']
#         self.dplh_model_handler.eval()

#         # Normalize the data
#         x_NN_scaled = transNorm(self.args, dataset_dict['x_nn'], varLst=self.args['var_t_nn'], toNorm=True)
#         c_NN_scaled = transNorm(self.args, dataset_dict['c_nn'], varLst=self.args['var_c_nn'], toNorm=True)
#         c_NN_scaled = np.repeat(np.expand_dims(c_NN_scaled, 0), x_NN_scaled.shape[0], axis=0)
#         dataset_dict['inputs_nn_scaled'] = np.concatenate((x_NN_scaled, c_NN_scaled), axis=2)
#         del x_NN_scaled, dataset_dict['x_nn']

#         # Convert numpy arrays to torch tensors
#         for key in dataset_dict.keys():
#             if type(dataset_dict[key]) == np.ndarray:
#                 dataset_dict[key] = torch.from_numpy(dataset_dict[key]).float()

#         self.config['batch_size'] = self.config['no_basins']
#         nt, ngrid, nx = dataset_dict['inputs_NN_scaled'].shape
#         rho = self.config['rho']

#         batch_size = self.config['batch_size']
#         iS = np.arange(0, ngrid, batch_size)
#         iE = np.append(iS[1:], ngrid)
#         list_out_diff_model = []

#         self.composite_sf = 0

#         for i in range(0, len(iS)):
#             dataset_dictionary_sample = take_sample_test(self.args, dataset_dict, iS[i], iE[i])
#             out_diff_model = self.dplh_model_handler(dataset_dictionary_sample)
#             out_lstm = self.ensemble_lstm(dataset_dictionary_sample)

#             # Calculate ensembled streamflow.
#             for mod in range(self.weights.shape[2]):
#                 self.composite_sf += self.ensemble_lstm.weights[:, :, mod] * out_diff_model[:, :, mod]
#             out_diff_model_cpu = {key: tensor.cpu().detach() for key, tensor in out_diff_model.items()}
#             list_out_diff_model.append(out_diff_model_cpu)

#         y_obs = dataset_dict['obs'][warm_up:, :, :]
#         save_output(self.args, list_out_diff_model, y_obs, calculate_metrics=True)
#         torch.cuda.empty_cache()












from gettext import NullTranslations
import logging
import os
from re import I
import time

import numpy as np
import torch
import tqdm
from conf.config import Config
from data.load_data.data_prep import No_iter_nt_ngrid, take_sample_train
from data.load_data.dataFrame_loading import loadData
from data.load_data.normalizing import init_norm_stats, transNorm
from data.utils.Dates import Dates
from models.multimodels.ensemble_network import EnsembleWeights
from models.multimodels.multimodel_handler import MultimodelHandler
from utils.utils import set_globals
from data.load_data.data_prep import take_sample_test
from utils.master import save_output

log = logging.getLogger(__name__)



class TestModel:
    """
    High-level multimodel testing handler; retrieves and formats testing data,
    initializes all individual models, and runs testing.
    """
    def __init__(self, config: Config):
        self.config = config
        self.config['device'], self.config['dtype'] = set_globals()

        # Initializing collection of differentiable hydrology models and their optimizers.
        # Training this object will parallel train all hydro models specified for ensemble.
        self.dplh_model_handler = MultimodelHandler(self.config).to(self.config['device'])
        # Initialize the weighting LSTM.
        self.ensemble_lstm = EnsembleWeights(self.config).to(self.config['device'])

    def _get_data_dict(self):
        log.info(f"Collecting testing data")

        # Preparing training data.
        # Formatting date ranges:
        self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
        self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
        self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

        # Read data for the test time range
        dataset_dict = loadData(self.config, trange=self.test_trange)
        
        warm_up = self.config['warm_up']
        nmul = self.config['nmul']
        self.dplh_model_handler.eval()

        # # Normalizations
        # # Stats for normalization of nn inputs:
        # init_norm_stats(self.config, dataset_dict['x_nn'], dataset_dict['c_nn'], dataset_dict['obs'])
        
        x_nn_scaled = transNorm(self.config, dataset_dict['x_nn'], varLst=self.config['observations']['var_t_nn'], toNorm=True)
        c_nn_scaled = transNorm(self.config, dataset_dict['c_nn'], varLst=self.config['observations']['var_c_nn'], toNorm=True)
        c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)
        dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
        del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
        self.dataset_dict = dataset_dict

        # Convert numpy arrays to torch tensors.
        for key in dataset_dict.keys():
            if type(dataset_dict[key]) == np.ndarray:
                dataset_dict[key] = torch.from_numpy(dataset_dict[key]).float()
                
        # Convert numpy arrays to torch tensors
        for key in dataset_dict.keys():
            if type(dataset_dict[key]) == np.ndarray:
                dataset_dict[key] = torch.from_numpy(dataset_dict[key]).float()
        self.dataset_dict = dataset_dict

        self.config['batch_size'] = self.config['n_basins']
        nt, ngrid, nx = dataset_dict['inputs_nn_scaled'].shape
        rho = self.config['rho']

        batch_size = self.config['batch_size']
        self.iS = np.arange(0, ngrid, batch_size)
        self.iE = np.append(self.iS[1:], ngrid)

    def run(self, experiment_tracker) -> None:
        log.info(f"Testing model: {self.config['name']}")

        self._get_data_dict()

        batch_preds_list = []
        for i in range(0, len(self.iS)):
            dataset_dict_sample = take_sample_test(self.config,
                                                         self.dataset_dict,
                                                         self.iS[i],
                                                         self.iE[i])
            model_preds = self.dplh_model_handler(dataset_dict_sample, eval=True)        
            self.ensemble_lstm(dataset_dict_sample, eval=True)
            wt_nn_preds = self.ensemble_lstm.weights_dict

            # print(model_preds['HBV']['flow_sim'].squeeze().shape)
            # print(wts_nn_preds['HBV'].shape)

            # print(self.dataset_dict['obs'][self.config['warm_up']:, :, :].shape)

            preds = 0
            # Calculate ensembled streamflow.
            for mod in self.config['hydro_models']:
                preds += wt_nn_preds[mod] * model_preds[mod]['flow_sim'].squeeze() # Linear combination of streamflows.
            
            batch_preds_list.append(preds.cpu().detach())
            print(batch_preds_list[0].shape)

        y_obs = self.dataset_dict['obs'][self.config['warm_up']:, :, :].squeeze()
        save_output(self.config, batch_preds_list, y_obs, calculate_metrics=True)
        
        torch.cuda.empty_cache()
