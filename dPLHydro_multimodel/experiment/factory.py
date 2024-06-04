import logging
from pathlib import Path
from typing import Dict

from conf.config import Config
# from data.all_edges_dataset import AllEdgesDataset
from dPLHydro_multimodel.data.general_dataset import GeneralDataset
from data.temporally_batched_dataset import TemporallyBatchedDataset
from data.utils import determine_proc_zone, format_gage_data
from dPLHydro_multimodel.utils.Dates import Dates
from data.utils.Dropout import Dropout
from injector import Module, multiprovider, provider
from models.hydro_models import PhysicsModel
from models.neural_networks import NeuralNetwork

log = logging.getLogger(__name__)


class Factory(Module):
    def __init__(self, cfg: Config):
        from data.utils import (read_gage_info, set_attributes,
                                set_global_indexing, set_min_max_statistics)
        from data.utils.ObservationReader import get_observation_reader

        self._observation_reader = get_observation_reader(cfg)
        gage_dict = read_gage_info(Path(cfg.observations.gage_info))
        attributes = set_attributes(cfg)
        (
            global_to_zone_mapping,
            zone_to_global_mapping,
        ) = set_global_indexing(attributes)
        attribute_statistics = set_min_max_statistics(cfg, attributes)
        self.dataset_inputs = {
            "attributes": attributes,
            "attribute_statistics": attribute_statistics,
            "global_to_zone_mapping": global_to_zone_mapping,
            "gage_dict": gage_dict,
            "zone_to_global_mapping": zone_to_global_mapping,
        }

    @provider
    def provide_train_dataset(self, cfg: Config) -> GeneralDataset:
        self.dataset_inputs["cfg"] = cfg
        self.dataset_inputs["dates"] = Dates(**cfg.train.model_dump())
        self.dataset_inputs["observations"] = self._observation_reader(
            cfg=cfg,
            dates=self.dataset_inputs["dates"],
            gage_dict=self.dataset_inputs["gage_dict"],
        ).read_data()
        self.dataset_inputs["dropout"] = Dropout(**cfg.train.model_dump())
        data = GeneralDataset(**self.dataset_inputs)
        return data

    @provider
    def provide_test_dataset(self, cfg: Config) -> TemporallyBatchedDataset:
        self.dataset_inputs["cfg"] = cfg
        self.dataset_inputs["dates"] = Dates(**cfg.test.model_dump())
        self.dataset_inputs["observations"] = self._observation_reader(
            cfg=cfg,
            dates=self.dataset_inputs["dates"],
            gage_dict=self.dataset_inputs["gage_dict"],
        ).read_data()
        self.dataset_inputs["data"] = format_gage_data(
            cfg=cfg, gage_dict=self.dataset_inputs["gage_dict"]
        )
        cfg = determine_proc_zone(cfg, self.dataset_inputs["data"])
        self.dataset_inputs["dropout"] = Dropout(**cfg.test.model_dump())
        data = TemporallyBatchedDataset(**self.dataset_inputs)
        return data

    # @provider
    # def provide_simulation_dataset(self, cfg: Config) -> AllEdgesDataset:
    #     self.dataset_inputs["cfg"] = cfg
    #     self.dataset_inputs["dates"] = Dates(**cfg.simulation.model_dump())
    #     self.dataset_inputs["observations"] = None
    #     self.dataset_inputs["dropout"] = None
    #     data = AllEdgesDataset(**self.dataset_inputs)
    #     return data

    @multiprovider
    def provide_network(self, cfg: Config) -> Dict[str, NeuralNetwork]:
        from models.neural_networks.cuda_mlp import MLP

        networks = {"mlp": MLP(cfg=cfg.spatial_mlp)}
        return networks

    @multiprovider
    def provide_model(self, cfg: Config) -> Dict[str, PhysicsModel]:
        from hydro_models import dMC
        from physics_models.saved_streamflow import MeritReader

        physics_models = {
            "streamflow": MeritReader(cfg=cfg),
            "dMC": dMC(cfg=cfg),
        }
        return physics_models
