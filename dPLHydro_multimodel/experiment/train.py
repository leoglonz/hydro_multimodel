import os
import logging

import torch
import numpy as np
import time
import tqdm

from conf.config import Config
from models.differentiable_model import dPLHydroModel
from models.multimodels.multimodel_handler import MultimodelHandler
from data.utils.Dates import Dates
from utils.utils import set_globals
from models.loss_functions.get_loss_function import get_loss_func
from data.load_data.normalizing import init_norm_stats, transNorm
from data.load_data.dataFrame_loading import loadData
from data.load_data.data_prep import (
    No_iter_nt_ngrid,
    take_sample_train,
)

log = logging.getLogger(__name__)



class TrainModel:
    """
    High-level multimodel training handler; retrieves and formats training data,
    initializes all individual models, sets optimizer, and runs training.
    """
    def __init__(self, config: Config):
        self.config = config
        self.config['device'], self.config['dtype'] = set_globals()
        self.init_models()

    def init_models(self):
        """
        Initialize and store each differentiable hydro model and optimizer in
        the ensemble.
        """
        self.model_dict = dict()
        self.optim_dict = dict()

        for mod in self.config['hydro_models']:
            self.model_dict[mod] = dPLHydroModel(self.config, mod).to(self.config['device'])
            self.optim_dict[mod] = torch.optim.Adadelta(self.model_dict[mod].parameters())

    def run(self, experiment_tracker) -> None:
        log.info(f"Training model: {self.config['name']}")

        # Preparing training data.
        # Formatting date ranges:
        self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
        self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
        self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

        dataset_dict = loadData(self.config, trange=self.train_trange)

        # Normalizations
        # Stats for normalization of nn inputs:
        init_norm_stats(self.config, dataset_dict['x_nn'], dataset_dict['c_nn'], dataset_dict['obs'])

        x_nn_scaled = transNorm(self.config, dataset_dict['x_nn'], varLst=self.config['var_t_nn'], toNorm=True)
        c_nn_scaled = transNorm(self.config, dataset_dict['c_nn'], varLst=self.config['var_c_nn'], toNorm=True)
        c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)
        del dataset_dict['x_nn'],   # no need the real values anymore
        dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
        del x_nn_scaled, c_nn_scaled   # we just need 'inputs_nn_model' which is a combination of these two

        # Initialize loss function.
        self.loss_func = get_loss_func(self.config, dataset_dict['obs']).to(self.config['device'])

        ngrid_train, nIterEp, nt, batchSize = No_iter_nt_ngrid(self.train_trange, self.config, dataset_dict['inputs_nn_scaled'])
        
        for mod in self.model_dict:
            self.model_dict[mod].zero_grad()
            self.model_dict[mod].train()

        # Use this to later implement code to run from checkpoint file
        start_epoch = 1
        for epoch in range(start_epoch, self.config['epochs'] + 1):
            t0 = time.time()
            prog_str = 'Epoch ' + str(epoch) + '/' + str(self.config['epochs'])

            for iIter in tqdm.tqdm(range(1, nIterEp + 1), desc=prog_str, leave=False, dynamic_ncols=True):
                dataset_dict_sample = take_sample_train(self.config, dataset_dict, ngrid_train, nt, batchSize)

                # Batch running of the differentiable models in parallel
                out_model_dict = dict()
                # Store loss across epochs, init to 0.
                ep_loss_dict = dict.fromkeys(self.model_dict, 0)

                for mod in self.model_dict:
                    # Forward each diff hydro model.
                    out_model_dict[mod] = self.model_dict[mod](dataset_dict_sample)

                    loss = self.loss_func(
                        self.config,
                        out_model_dict[mod],
                        dataset_dict_sample['obs'],
                        igrid=dataset_dict_sample['iGrid']
                        )
                    
                    loss.backward()
                    self.optim_dict[mod].step()
                    self.model_dict[mod].zero_grad()
                    ep_loss_dict[mod] = ep_loss_dict[mod] + loss.item()

            ep_loss_dict['HBV'] = ep_loss_dict['HBV'] / nIterEp
            logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory".format(
                epoch, ep_loss_dict['HBV'], time.time() - t0, int(torch.cuda.memory_allocated(device=self.config["device"]) * 0.001))
            print(logStr)

            if epoch % self.config["save_epoch"] == 0:
                # Save models
                for mod in self.model_dict:                
                    save_dir = os.path.join(self.config["out_dir"], mod+ "_model_Ep" + str(epoch) + ".pt")
                    torch.save(self.model_dict[mod], save_dir)
