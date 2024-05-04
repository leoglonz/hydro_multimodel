from gettext import NullTranslations
import logging
import os
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

log = logging.getLogger(__name__)



class TrainWeightsModel:
    """
    High-level multimodel training handler; injests pretrained differentiable hydrology models and trains a weighting
    LSTM to dynamically join their oututs.
    """
    def __init__(self, config: Config):
        self.config = config
        self.config['device'], self.config['dtype'] = set_globals()

        # Initializing collection of trained differentiable hydrology models.
        self.dplh_model_handler = MultimodelHandler(self.config).to(self.config['device'])
        # Initialize the weighting LSTM.
        self.ensemble_lstm = EnsembleWeights(self.config).to(self.config['device'])


    def _get_data_dict(self) -> None:
        log.info(f"Collecting training data")

        # Preparing training data.
        # Formatting date ranges:
        self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
        self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
        self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

        dataset_dict = loadData(self.config, trange=self.train_trange)

        # Normalizations
        # Stats for normalization of nn inputs:
        init_norm_stats(self.config, dataset_dict['x_nn'], dataset_dict['c_nn'], dataset_dict['obs'])
        
        x_nn_scaled = transNorm(self.config, dataset_dict['x_nn'], varLst=self.config['observations']['var_t_nn'], toNorm=True)
        c_nn_scaled = transNorm(self.config, dataset_dict['c_nn'], varLst=self.config['observations']['var_c_nn'], toNorm=True)
        c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)
        del dataset_dict['x_nn']
        dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
        del x_nn_scaled, c_nn_scaled   # we just need 'inputs_nn_model' which is a combination of these two.

        self.dataset_dict = dataset_dict


    def run(self, experiment_tracker) -> None:
        log.info(f"Training model: {self.config['name']}")

        self._get_data_dict()

        ngrid_train, minibatch_iter, nt, batch_size = No_iter_nt_ngrid(self.train_trange,
                                                               self.config,
                                                               self.dataset_dict['inputs_nn_scaled'])
    
        # Initialize the loss function and optimizer:
        self.ensemble_lstm.init_loss_func(self.dataset_dict['obs'])









        # Initialize loss function(s) and optimizer.
        self.dplh_model_handler.init_loss_func(self.dataset_dict['obs'])
        optim = self.dplh_model_handler.optim

        if self.config['ensemble_type'] != None:
            self.ensemble_lstm.init_loss_func(self.dataset_dict['obs'])
            optim.add_param_group({'params': self.ensemble_lstm.model_params})


        start_epoch = 1  # Use this to later implement code to run from checkpoint file
        for epoch in range(start_epoch, self.config['epochs'] + 1):
            # Store loss across epochs, init to 0.
            ep_loss_dict = dict.fromkeys(self.config['hydro_models'], 0)
            if self.config['ensemble_type'] != None:
                ep_loss_dict['wtNN'] = 0

            start_time = time.perf_counter()
            prog_str = 'Epoch ' + str(epoch) + '/' + str(self.config['epochs'])

            for i_iter in tqdm.tqdm(range(1, minibatch_iter + 1), desc=prog_str, leave=False, dynamic_ncols=True):
                dataset_dict_sample = take_sample_train(self.config,
                                                        self.dataset_dict,
                                                        ngrid_train,
                                                        nt,
                                                        batch_size)

                # Forward diff hydro models.
                self.model_preds = self.dplh_model_handler(dataset_dict_sample)

                # Epoch loss for all diff hydro models.
                hydro_loss, ep_loss_dict = self.dplh_model_handler.calc_loss(ep_loss_dict)

                if (self.config['ensemble_type'] != None) and (self.config['freeze_para_nn'] == False):
                        # Train weighting network in parallel w/ diff hydro models.
                        self.ensemble_lstm(dataset_dict_sample)

                        # Compute loss.
                        wt_nn_loss, ep_loss_dict = self.ensemble_lstm.calc_loss(self.model_preds, ep_loss_dict)
                else:
                    wt_nn_loss = 0

                # Run backward and optimizer on all models simultaneously.
                total_loss = hydro_loss + wt_nn_loss
                total_loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)  # set none avoids costly read-writes. Could be problematic.


            # Log epoch stats.
            ep_loss_dict = {key: value / minibatch_iter for key, value in ep_loss_dict.items()}
            loss_formated = ", ".join(f"{key}: {value:.6f}" for key, value in ep_loss_dict.items())
            elapsed = time.perf_counter() - start_time
            mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
            log.info("Per-model loss after epoch {}: {} \n".format(epoch,loss_formated) +
                    "~ Runtime {:.2f} sec, {} Mb reserved GPU memory".format(elapsed,mem_aloc))
            
            # Save models:
            if epoch % self.config['save_epoch'] == 0:
                for mod in self.config['hydro_models']:                
                    save_dir = os.path.join(self.config['output_dir'], mod+ '_model_Ep' + str(epoch) + '.pt')
                    torch.save(self.dplh_model_handler.model_dict[mod], save_dir)

        if self.config['ensemble_type'] != None:
            if self.config['freeze_para_nn'] == True:
                # Train weighting network after hydro models have been trained
                # and their parameterization networks have been frozen.
                self.minibatch_iter = minibatch_iter
                self.ngrid_train = ngrid_train
                self.nt = nt
                self.batch_size = batch_size
                self.run_ensemble_train()            
    
    def run_ensemble_train(self):
        """
        Only used when training parameterization and weighting networks in series
        (i.e., training the weighting network with parameterization networks frozen).
        """
        # Use this to later implement code to run from checkpoint file
        start_epoch = 1
        for epoch in range(start_epoch, self.config['epochs'] + 1):
            wt_loss = 0

            start_time = time.perf_counter()
            prog_str = 'Epoch ' + str(epoch) + '/' + str(self.config['epochs'])

            for i_iter in tqdm.tqdm(range(1, self.minibatch_iter + 1), desc=prog_str, leave=False, dynamic_ncols=True):
                dataset_dict_sample = take_sample_train(self.config,
                                                        self.dataset_dict,
                                                        self.ngrid_train,
                                                        self.nt,
                                                        self.batch_size)

                # Train weighting network in parallel w/ diff hydro models.
                self.ensemble_lstm(dataset_dict_sample)

                # Compute loss.
                self.ensemble_lstm.calc_loss(self.model_preds, wt_loss)
