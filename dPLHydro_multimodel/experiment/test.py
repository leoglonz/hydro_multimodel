import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from conf.config import Config
from experiment.experiment_tracker import Tracker
from injector import inject
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class TestModel:
    """
    High-level multimodel testing handler; retrieves and formats testing data,
    initializes all individual models, and runs testing.
    """
    def __init__(self, config: Config, diff_models):
        self.config = config
        self.diff_models = diff_models

    def run(self):
        warm_up = self.args["warm_up"]
        nmul = self.args["nmul"]
        self.diff_model.eval()

        # Read data for the test time range
        dataset_dictionary = loadData(self.args, trange=self.args["t_test"])

        # Normalize the data
        x_NN_scaled = transNorm(self.args, dataset_dictionary["x_NN"], varLst=self.args["varT_NN"], toNorm=True)
        c_NN_scaled = transNorm(self.args, dataset_dictionary["c_NN"], varLst=self.args["varC_NN"], toNorm=True)
        c_NN_scaled = np.repeat(np.expand_dims(c_NN_scaled, 0), x_NN_scaled.shape[0], axis=0)
        dataset_dictionary["inputs_NN_scaled"] = np.concatenate((x_NN_scaled, c_NN_scaled), axis=2)
        del x_NN_scaled, dataset_dictionary["x_NN"]

        # Convert numpy arrays to torch tensors
        for key in dataset_dictionary.keys():
            if type(dataset_dictionary[key]) == np.ndarray:
                dataset_dictionary[key] = torch.from_numpy(dataset_dictionary[key]).float()

        args["batch_size"] = args["no_basins"]
        nt, ngrid, nx = dataset_dictionary["inputs_NN_scaled"].shape
        rho = args["rho"]

        batch_size = args["batch_size"]
        iS = np.arange(0, ngrid, batch_size)
        iE = np.append(iS[1:], ngrid)
        list_out_diff_model = []

        for i in range(0, len(iS)):
            dataset_dictionary_sample = take_sample_test(self.args, dataset_dictionary, iS[i], iE[i])
            out_diff_model = self.diff_model(dataset_dictionary_sample)
            out_diff_model_cpu = {key: tensor.cpu().detach() for key, tensor in out_diff_model.items()}
            list_out_diff_model.append(out_diff_model_cpu)

        y_obs = dataset_dictionary["obs"][warm_up:, :, :]
        save_outputs(self.args, list_out_diff_model, y_obs, calculate_metrics=True)
        torch.cuda.empty_cache()
