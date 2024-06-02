import logging
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from conf.config import Config
from data.load_data.data_prep import (No_iter_nt_ngrid, take_sample_test,
                                      take_sample_train)
from data.load_data.dataFrame_loading import loadData
from data.load_data.normalizing import init_norm_stats, transNorm
from data.utils.Dates import Dates
from models.multimodels.ensemble_network import EnsembleWeights
from models.multimodels.multimodel_handler import MultimodelHandler
from utils.master import save_outputs
from utils.stat import statError
from utils.utils import set_globals

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
        if self.config['ensemble_type'] != 'None':
            self.ensemble_lstm = EnsembleWeights(self.config).to(self.config['device'])

    def _get_data_dict(self):
        log.info(f"Collecting testing data")

        # Prepare training data.
        self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
        self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
        self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

        # Read data for the test time range
        dataset_dict = loadData(self.config, trange=self.test_trange)

        # Normalizatio ns
        # init_norm_stats(self.config, dataset_dict['x_nn'], dataset_dict['c_nn'], dataset_dict['obs'])
        x_nn_scaled = transNorm(self.config, dataset_dict['x_nn'], varLst=self.config['observations']['var_t_nn'], toNorm=True)
        c_nn_scaled = transNorm(self.config, dataset_dict['c_nn'], varLst=self.config['observations']['var_c_nn'], toNorm=True)
        c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)
        dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
        del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
        
        # Convert numpy arrays to torch tensors
        for key in dataset_dict.keys():
            if type(dataset_dict[key]) == np.ndarray:
                dataset_dict[key] = torch.from_numpy(dataset_dict[key]).float()
        self.dataset_dict = dataset_dict

        ngrid = dataset_dict['inputs_nn_scaled'].shape[1]
        self.iS = np.arange(0, ngrid, self.config['n_basins'])
        self.iE = np.append(self.iS[1:], ngrid)

    def run(self, experiment_tracker) -> None:
        log.info(f"Testing model: {self.config['name']}")

        self._get_data_dict()
        # self.dplh_model_handler.eval()
        
        # Get model predictions.
        batched_preds_list = []
        for i in tqdm.tqdm(range(0, len(self.iS)),
                           desc=f"Testing on batches of {self.config['n_basins']}",
                           leave=False,
                           dynamic_ncols=True):
            
            dataset_dict_sample = take_sample_test(self.config,
                                                   self.dataset_dict,
                                                   self.iS[i],
                                                   self.iE[i])
            
            hydro_preds = self.dplh_model_handler(dataset_dict_sample, eval=True)

            if self.config['ensemble_type'] != 'None':
                # Calculate ensembled streamflow.
                wt_nn_preds = self.ensemble_lstm(dataset_dict_sample, eval=True)
                ensemble_pred = self.ensemble_lstm.ensemble_models(hydro_preds)

                # batched_preds_list.append(ensemble_pred.cpu().detach())
                batched_preds_list.append({key: tensor.cpu().detach() for key,
                                           tensor in ensemble_pred.items()})
            else:
                model_name = self.config['hydro_models'][0]
                batched_preds_list.append({key: tensor.cpu().detach() for key,
                                           tensor in hydro_preds[model_name].items()})

        # Get observation data.
        y_obs = self.dataset_dict['obs'][self.config['warm_up']:, :, :]

        save_outputs(self.config, batched_preds_list, y_obs)

        self.calc_metrics(batched_preds_list, y_obs)
        torch.cuda.empty_cache()

    def calc_metrics(self, batched_preds_list, y_obs):
        """
        Calculate and save model test metrics to csv.
        """
        preds_list = list()
        obs_list = list()
        name_list = []
        
        flow_sim = torch.cat([d['flow_sim'] for d in batched_preds_list], dim=1)
        flow_obs = y_obs[:, :, self.config['target'].index('00060_Mean')]
        preds_list.append(flow_sim.numpy())
        obs_list.append(np.expand_dims(flow_obs, 2))
        name_list.append('flow')
    
        # we need to swap axes here to have [basin, days]
        statDictLst = [
            statError(np.swapaxes(x.squeeze(), 1, 0), np.swapaxes(y.squeeze(), 1, 0))
            for (x, y) in zip(preds_list, obs_list)
        ]
        ### save this file
        # median and STD calculation
        for stat, name in zip(statDictLst, name_list):
            count = 0
            mdstd = np.zeros([len(stat), 3])
            for key in stat.keys():
                median = np.nanmedian(stat[key])  # abs(i)
                STD = np.nanstd(stat[key])  # abs(i)
                mean = np.nanmean(stat[key])  # abs(i)
                k = np.array([[median, STD, mean]])
                mdstd[count] = k
                count = count + 1
            mdstd = pd.DataFrame(
                mdstd, index=stat.keys(), columns=['median', 'STD', 'mean']
            )

            mdstd.to_csv((os.path.join(self.config['testing_dir'], 'mdstd_' + name + '.csv')))