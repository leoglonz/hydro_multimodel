"""
Use this script to run multimodel training/testing.
"""

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Union

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from conf.config import Config, ModeEnum
# from dMC.experiment_handler import build_handler
# from dMC.experiment_handler.experiment_tracker import ExperimentTracker

log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="conf/",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    try:
        start_time = time.perf_counter()

        config = initialize_config(cfg)
    #     experiment_tracker = ExperimentTracker(cfg=config)

        experiment_name = config.mode
    #     log.info(f"USING MODE: {config.mode}")

    #     if config.mode == ModeEnum.train_test:
    #         # Train the model
    #         config.mode = ModeEnum.train
    #         train_experiment_handler = build_handler(config)
    #         train_experiment_handler.run(config, experiment_tracker)

    #         # Make sure the model weights are transfered to the new handler
    #         config.mode = ModeEnum.test
    #         test_experiment_handler = build_handler(config)
    #         test_experiment_handler.neural_networks = (
    #             train_experiment_handler.neural_networks
    #         )
    #         dist.barrier()

    #         # Test the model
    #         test_experiment_handler.run(config, experiment_tracker)
    #         dist.barrier()
    #     else:
    #         # Run any other experiment
    #         experiment_handler = build_handler(config)
    #         experiment_handler.run(config, experiment_tracker)

        total_time = time.perf_counter() - start_time
        log.info(
            f"| {experiment_name} completed | "
            f"Time Elapsed : {(total_time / 60):.6f} minutes"
        ) 
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")


def initialize_config(cfg: DictConfig) -> Config:
    try:
        # Convert the DictConfig to a dictionary and then to a Config object for validation
        config_dict: Union[Dict[str, Any], Any] = OmegaConf.to_container(
            cfg, resolve=True
        )
        config = Config(**config_dict)
    except ValidationError as e:
        log.exception(e)
        raise e
    # if config_dict["local_rank"] == 0:
    #     _save_cfg(cfg=config)
    # _set_seed(cfg=config)
    # _set_device(cfg=config)
    return config


# def _save_cfg(cfg: Config) -> None:
#     import warnings

#     warnings.filterwarnings(
#         action="ignore",
#         category=UserWarning,
#         message=r"^Pydantic serializer warnings:\n.*Expected `str` but got `PosixPath`.*",
#     )
#     save_path = Path() / "pydantic_config.yaml"
#     json_cfg = cfg.model_dump_json(indent=4)
#     log.info(f"Running the following config:\n{json_cfg}")

#     with save_path.open("w") as f:
#         OmegaConf.save(config=OmegaConf.create(json_cfg), f=f)


# def _set_seed(cfg: Config) -> None:
#     torch.manual_seed(cfg.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(cfg.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     np.random.seed(cfg.np_seed)
#     random.seed(cfg.seed)


# def _set_device(cfg: Config) -> None:
#     rank = cfg.local_rank
#     device = cfg.device[rank]
#     torch.cuda.set_device(device=device)



if __name__ == "__main__":
    main()
