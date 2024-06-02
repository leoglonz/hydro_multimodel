import logging

from conf.config import Config, ModeEnum
from experiment.test import TestModel
from experiment.train import TrainModel
from experiment.train_wts_only import TrainWeightsModel

# from experiment.factory import Factory
# from experiment.test_handler import TestHandler
# from experiment.train_handler import TrainHandler

# from typing import Union

# from injector import Injector, singleton


log = logging.getLogger(__name__)



def build_handler(cfg: Config, config_dict: dict): #-> Union[TrainHandler, TestHandler]:
    # injector = Injector([configure(cfg), Factory(cfg)])
    if cfg.mode == ModeEnum.train:
        # return injector.get(TrainHandler)
        return TrainModel(config_dict)
    elif cfg.mode == ModeEnum.test:
        # return injector.get(TestHandler)
        return TestModel(config_dict)
    elif cfg.mode == ModeEnum.train_wts_only:
        return TrainWeightsModel(config_dict)
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


# def configure(cfg: Config):
#     def _bind(binder):
#         """
#         Binds the Configuration to a singleton scope.

#         :param binder: Binder object.
#         """
#         binder.bind(Config, to=cfg, scope=singleton)

#     return _bind
