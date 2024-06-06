"""
Use this script interfaces with experiment (train/test) scripts.
"""

import logging
import time
from typing import Any, Dict, Union

import hydra
import torch
from conf.config import Config, ModeEnum
from core.utils import (create_output_dirs, randomseed_config, set_system_spec,
                        show_args)
from experiment import build_handler
from experiment.experiment_tracker import ExperimentTracker
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

log = logging.getLogger(__name__)



@hydra.main(
    version_base="1.3",
    config_path="conf/",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    try:
        start_time = time.perf_counter()

        # Injest config yaml.
        ## Temporarily used config dictionary until validation code is done.
        config, config_dict = initialize_config(cfg)
        experiment_tracker = ExperimentTracker(cfg=config)

        # Set device, dtype.
        config.device, config.dtype = set_system_spec(config.gpu_id)
        config_dict = create_output_dirs(config_dict)

        experiment_name = config.mode
        log.info(f"RUNNING MODE: {config.mode}")
        show_args(config)

        if config.mode == ModeEnum.train_test:
            # Run training and testing together.
            # Train:
            config.mode = ModeEnum.train
            train_experiment_handler = build_handler(config, config_dict)
            train_experiment_handler.run(experiment_tracker=experiment_tracker)

            # Test: (first transfer weights)
            config.mode = ModeEnum.test
            test_experiment_handler = build_handler(config, config_dict)            
            test_experiment_handler.dplh_model_handler = train_experiment_handler.dplh_model_handler
            if config_dict['ensemble_type'] != 'None':
                test_experiment_handler.ensemble_lstm = train_experiment_handler.dplh_model_handler

            test_experiment_handler.run(experiment_tracker=experiment_tracker)

        else:
            # Run either training or testing. 
            experiment_handler = build_handler(config, config_dict)
            experiment_handler.run(experiment_tracker=experiment_tracker)

        total_time = time.perf_counter() - start_time
        log.info(
            f"| {experiment_name} completed | "
            f"Time Elapsed : {(total_time / 60):.6f} minutes"
        ) 

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")
        torch.cuda.empty_cache()


def initialize_config(cfg: DictConfig) -> Config:
    """
    Convert config into a dictionary, and a Config object for validation.
    """
    try:
        config_dict: Union[Dict[str, Any], Any] = OmegaConf.to_container(
            cfg, resolve=True
        )
        config = Config(**config_dict)
    except ValidationError as e:
        log.exception(e)
        raise e
   
    return config, config_dict



if __name__ == "__main__":
    randomseed_config()
    main()
    print("Experiment ended.")
