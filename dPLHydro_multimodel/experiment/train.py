"""
Train a pytorch model.
"""
import logging
import time

import numpy as np
from sympy import S
import torch
import tqdm
from typing import Dict, Any, Tuple

from conf.config import Config
from core.calc.normalize import init_norm_stats, trans_norm
from core.data import no_iter_nt_ngrid, take_sample_train
from core.data.dataFrame_loading import load_data
from core.utils import save_model
from core.utils.Dates import Dates
from models.model_handler import ModelHandler
from models.multimodels.ensemble_network import EnsembleWeights

log = logging.getLogger(__name__)



class TrainModel:
    """
    High-level multimodel training handler; retrieves and formats training data,
    initializes all individual models, sets optimizer, and runs training.
    """
    def __init__(self, config: Config) -> None:
        self.config = config

        # Initialize collection of differentiable hydrology models and their optimizers.
        # NOTE: training this object parallel trains all models in ensemble.
        self.dplh_model_handler = ModelHandler(self.config).to(self.config['device'])
        
        # Initialize weighting LSTM (wNN) if ensemble type is specified.
        if self.config['ensemble_type'] != 'none':
            self.ensemble_lstm = EnsembleWeights(self.config).to(self.config['device'])

    def _get_data_dict(self) -> None:
        log.info(f"Collecting training data")

        # Prepare training data.
        self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
        self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
        self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

        dataset_dict = load_data(self.config)

        # Normalizations
        init_norm_stats(self.config, dataset_dict['x_nn'], dataset_dict['c_nn'],
                        dataset_dict['obs'])
        x_nn_scaled = trans_norm(self.config, dataset_dict['x_nn'],
                                 varLst=self.config['observations']['var_t_nn'], toNorm=True)
        c_nn_scaled = trans_norm(self.config, dataset_dict['c_nn'],
                                 varLst=self.config['observations']['var_c_nn'], toNorm=True)
        c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0),
                                x_nn_scaled.shape[0], axis=0)
        dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
        del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']

        self.dataset_dict = dataset_dict
    
    def run(self, experiment_tracker) -> None:
        """
        High-level management of ensemble/non-ensemble model training .
        """
        log.info(f"Training model: {self.config['name']}")

        self._get_data_dict()

        ngrid_train, minibatch_iter, nt, batch_size = no_iter_nt_ngrid(
            self.train_trange, self.config, self.dataset_dict['inputs_nn_scaled'])

        # Initialize loss function(s) and optimizer.
        self.dplh_model_handler.init_loss_func(self.dataset_dict['obs'])
        optim = self.dplh_model_handler.optim

        if self.config['ensemble_type'] != 'none':
            self.ensemble_lstm.init_loss_func(self.dataset_dict['obs'])
            optim.add_param_group({'params': self.ensemble_lstm.model_params})

        start_ep = self.config['checkpoint']['start_epoch'] if self.config['use_checkpoint'] else 1

        # Start of training.
        for epoch in range(start_ep, self.config['epochs'] + 1):
            log.info(f"Starting epoch {epoch}")
            start_time = time.perf_counter()

            self._train_epoch(epoch, minibatch_iter, ngrid_train, nt, batch_size, optim)
            self._log_epoch_stats(epoch, self.ep_loss_dict, minibatch_iter, start_time)

            if epoch % self.config['save_epoch'] == 0:
                self.save_models(epoch)

        if self.config['ensemble_type'] == 'frozen_pnn':
            # For ensembles: Train weighting network (wNN)
            self.run_ensemble_frozen(ngrid_train, minibatch_iter, nt, batch_size, start_ep)

    def _train_epoch(self, epoch: int, minibatch_iter: int, ngrid_train: Any, nt: int,
                     batch_size: int, optim: torch.optim.Optimizer) -> None:
        # Allows storing loss for multiple models.
        # ep_loss_dict = dict.fromkeys(self.config['hydro_models'], 0)
        ep_loss_dict = {key: 0 for key in self.config['hydro_models']}
        if self.config['ensemble_type'] != 'none':
            ep_loss_dict['wtNN'] = 0

        prog_str = f"Epoch {epoch}/{self.config['epochs']}"

        for i in tqdm.tqdm(range(1, minibatch_iter + 1), desc=prog_str,
                           leave=False, dynamic_ncols=True):
            dataset_dict_sample = take_sample_train(self.config, self.dataset_dict,
                                                    ngrid_train, nt, batch_size)

            # Forward pass for hydrology models.
            model_preds = self.dplh_model_handler(dataset_dict_sample)
            hydro_loss, ep_loss_dict = self.dplh_model_handler.calc_loss(ep_loss_dict)

            if self.config['ensemble_type'] == 'free_pnn':
                # Forward pass for weighting network (wNN).
                self.ensemble_lstm(dataset_dict_sample)
                wnn_loss, ep_loss_dict = self.ensemble_lstm.calc_loss(model_preds, ep_loss_dict)
            else:
                wnn_loss = 0
            
            self.ep_loss_dict = ep_loss_dict

            total_loss = hydro_loss + wnn_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad() # set_to_none=True actually increases runtimes.

    def _log_epoch_stats(self, epoch: int, ep_loss_dict: Dict[str, float],
                            minibatch_iter: int, start_time: float) -> None:
        ep_loss_dict = {key: value / minibatch_iter for key, value in ep_loss_dict.items()}
        loss_formated = ", ".join(f"{key}: {value:.6f}" for key, value in ep_loss_dict.items())
        elapsed = time.perf_counter() - start_time
        mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
        log.info(f"Model loss after epoch {epoch}: {loss_formated} \n~ Runtime {elapsed:.2f} sec,{mem_aloc} Mb reserved GPU memory")

    def run_ensemble_frozen(self, ngrid_train: Any, minibatch_iter: int, nt: int,
                            batch_size: int, start_ep: int) -> None:
        """
        For ensemble models:
        Train the weighting network (wNN) after dPL hydrology models have been trained
        and their parameterization networks (pNN) have been frozen.
        """
        log.info("Freezing hydro model pNNs. Training wNN...")
        
        # Initialize a fresh optimizer.
        self.ensemble_lstm.init_loss_func(self.dataset_dict['obs'])
        self.ensemble_lstm.init_optimizer()
        optim = self.ensemble_lstm.optim

        for epoch in range(self.start_epoch, self.config['epochs'] + 1):
            start_time = time.perf_counter()
            prog_str = f"Epoch {epoch}/{self.config['epochs']}"
            ep_loss = {'wNN': 0}

            for i in tqdm.tqdm(range(1, minibatch_iter + 1), desc=prog_str,
                               leave=False, dynamic_ncols=True):
                dataset_dict_sample = take_sample_train(self.config, self.dataset_dict,
                                                        ngrid_train, nt, batch_size)

                # Forward pass
                model_preds = self.dplh_model_handler(dataset_dict_sample)
                self.ensemble_lstm(dataset_dict_sample)
                
                total_loss = self.ensemble_lstm.calc_loss(model_preds)
                total_loss.backward()
                optim.step()
                optim.zero_grad()
                ep_loss += total_loss.item()

            self._log_epoch_stats(epoch, ep_loss, minibatch_iter, start_time)

            # Save model
            if epoch % self.config['save_epoch'] == 0:
                self.save_models(epoch, frozen_pnn=True)

    def save_models(self, epoch: int, frozen_pnn: bool = False) -> None:
        """
        Save hydrology and/or weighting models.
        frozen_pnn flag specifies only saving weighting model (wNN) after pNNs
        have been frozen.
        """
        if frozen_pnn:
            save_model(self.config, self.ensemble_lstm.lstm, 'wNN_model', epoch)
        else:
            for mod in self.config['hydro_models']:
                save_model(self.config, self.dplh_model_handler.model_dict[mod], mod, epoch)

            if self.config['ensemble_type'] == 'free_pnn':
                save_model(self.config, self.ensemble_lstm.lstm, 'wNN_model', epoch)
                