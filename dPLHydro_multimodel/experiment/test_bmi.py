import logging
import os

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



class TestBMIModel:
    """
    High-level multimodel testing handler; retrieves and formats testindata,
    initializes all individual models, and runs testing.
    """
    def __init__(self, config: Config):
        raise NotImplementedError('__init__')

    def run(self, experiment_tracker) -> None:
        log.info(f"Testing model: {self.config['name']}")

        raise NotImplementedError('run')
    