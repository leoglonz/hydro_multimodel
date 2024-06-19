import logging
import os

from networkx import bidirectional_dijkstra
import numpy as np
import pandas as pd
import torch
import tqdm
from conf.config import Config
from core.calc.normalize import trans_norm
from core.calc.stat import stat_error
from core.data import take_sample_test
from core.data.dataFrame_loading import load_data
from core.utils import save_outputs
from core.utils.Dates import Dates
from models.model_handler import ModelHandler
from models.multimodels.ensemble_network import EnsembleWeights

from models.bmi.bmi_differentiable_model import BMIdPLHydroModel


log = logging.getLogger(__name__)

cfg_filepath = ''



class TestBMIModel:
    """
    High-level multimodel testing handler; retrieves and formats testindata,
    initializes all individual models, and runs testing.
    """
    def __init__(self):
        log.info("Initializing BMI forward instance.")

    def _get_data_dict(self):
        log.info(f"Collecting testing data")

        # Prepare training data.
        self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
        self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
        self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

        # Read data for the test time range
        dataset_dict = load_data(self.config, trange=self.test_trange)

        # Normalizations
        # init_norm_stats(self.config, dataset_dict['x_nn'], dataset_dict['c_nn'], dataset_dict['obs'])
        x_nn_scaled = trans_norm(self.config, dataset_dict['x_nn'], varLst=self.config['observations']['var_t_nn'], toNorm=True)
        c_nn_scaled = trans_norm(self.config, dataset_dict['c_nn'], varLst=self.config['observations']['var_c_nn'], toNorm=True)
        c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)
        dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
        del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
        
        # Convert numpy arrays to torch tensors
        for key in dataset_dict.keys():
            if type(dataset_dict[key]) == np.ndarray:
                dataset_dict[key] = torch.from_numpy(dataset_dict[key]).float()
        self.dataset_dict = dataset_dict

        ngrid = dataset_dict['inputs_nn_scaled'].shape[1]
        self.iS = np.arange(0, ngrid, self.config['batch_basins'])
        self.iE = np.append(self.iS[1:], ngrid)

    def run(self, experiment_tracker) -> None:
        log.info(f"Testing BMI model: {self.config['name']}")

        # Instantiate BMI object.
        log.info("Creating dPLHydro BMI object")
        model = BMIdPLHydroModel()

        # Initialize the BMI.
        log.info("Initializing BMI")
        model.initialize(bmi_cfg_filepath=cfg_filepath)

        # Extract data and get forcings, basin attributes.
        self._get_data_dict()
        forcings_dict = self.dataset_dict['inputs_nn_scaled']['x_nn_scaled']
        attributes_dict = self.dataset_dict['inputs_nn_scaled']['c_nn_scaled']

        # TODO: Set attributes to validated keys within model.
        model.set_value('attribute_x', attributes_dict['x'])

        # Run through all available forcings.
        # TODO: allow specification of update time range.
        n_forcings = forcings_dict.size[0]

        for day in range(n_forcings):
            # TODO: extract each forcing out from this and map to value with "set_value". The same should be done for the basin attributes.
            forcings = forcings[day,:,:]

            # Set forcings to validated keys within model.
            model.set_value('forcing_x', forcings['x'])
            model.set_value('forcing_y', forcings['y'])

            # 1-timestep update of BMI.
            model.update()

            if day > 10:
                # In case of runaway during debut
                log.info("Terminating for debug.")
                break

        # Finalization step for BMI
        log.info("Finalizing BMI and wrapping up...")
        model.finalize()
