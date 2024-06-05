import logging
import os
import time
import logging
import numpy as np
import torch
import tqdm
from conf.config import Config
from core.data import no_iter_nt_ngrid, take_sample_train
from core.data.dataFrame_loading import load_data
from core.calc.normalize import init_norm_stats, trans_norm
from core.utils.Dates import Dates
from models.multimodels.ensemble_network import EnsembleWeights
from models.multimodels.multimodel_handler import MultimodelHandler
from core.utils import save_model

log = logging.getLogger(__name__)



class TrainWeightsModel:
    """
    High-level multimodel training handler; injests pretrained differentiable hydrology models and trains a weighting
    LSTM to dynamically join their oututs.
    """
    def __init__(self, config: Config):
        self.config = config

        # Initializing collection of trained differentiable hydrology models and weighting LSTM.
        self.dplh_model_handler = MultimodelHandler(self.config).to(self.config['device'])
        self.ensemble_lstm = EnsembleWeights(self.config).to(self.config['device'])

    def _get_data_dict(self):
        log.info(f"Collecting training data")

        # Prepare training data,format date ranges:
        self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
        # self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
        # self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

        dataset_dict = load_data(self.config, trange=self.train_trange)

        # Normalization + stats for nn input normalization:
        init_norm_stats(self.config, dataset_dict['x_nn'], dataset_dict['c_nn'], dataset_dict['obs'])
        
        x_nn_scaled = trans_norm(self.config, dataset_dict['x_nn'], varLst=self.config['observations']['var_t_nn'], toNorm=True)
        c_nn_scaled = trans_norm(self.config, dataset_dict['c_nn'], varLst=self.config['observations']['var_c_nn'], toNorm=True)
        c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)
        dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
        del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
        self.dataset_dict = dataset_dict

    def run(self, experiment_tracker) -> None:
        log.info(f"Training model: {self.config['name']}")

        self._get_data_dict()

        ngrid_train, minibatch_iter, nt, batch_size = no_iter_nt_ngrid(self.train_trange,
                                                               self.config,
                                                               self.dataset_dict['inputs_nn_scaled'])
    
        # Initialize the loss function and optimizer:
        self.ensemble_lstm.init_loss_func(self.dataset_dict['obs'])
        self.ensemble_lstm.init_optimizer()
        optim = self.ensemble_lstm.optim

        if self.config['use_checkpoint'] == True:
            start_epoch = self.config['checkpoint']['start_epoch']
        else:
            start_epoch = 1

        for epoch in range(start_epoch, self.config['epochs'] + 1):
            ep_loss = 0

            start_time = time.perf_counter()
            prog_str = 'Epoch ' + str(epoch) + '/' + str(self.config['epochs'])

            for i_iter in tqdm.tqdm(range(1, minibatch_iter + 1), desc=prog_str, leave=False, dynamic_ncols=True):
                dataset_dict_sample = take_sample_train(self.config,
                                                        self.dataset_dict,
                                                        ngrid_train,
                                                        nt,
                                                        batch_size)

                # Forward diff hydro models and weighting network.
                self.model_preds = self.dplh_model_handler(dataset_dict_sample)
                self.ensemble_lstm(dataset_dict_sample)

                # Loss calculation + step optimizer.
                loss = self.ensemble_lstm.calc_loss(self.model_preds)
                ep_loss += loss.item()
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)  # Set none avoids costly read-writes, but could actually increase run times

            # Log epoch stats.
            elapsed = time.perf_counter() - start_time
            mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
            log.info("Weighting network loss after epoch {}: {:.6f} \n".format(epoch,ep_loss) +
                    "~ Runtime {:.2f} sec, {} Mb reserved GPU memory".format(elapsed,mem_aloc))
            
            # Save models:
            if epoch % self.config['save_epoch'] == 0:
                save_dir = os.path.join(self.config['output_dir'], 'wtNN_model_Ep' + str(epoch) + '.pt')
                torch.save(self.ensemble_lstm.lstm, save_dir)
