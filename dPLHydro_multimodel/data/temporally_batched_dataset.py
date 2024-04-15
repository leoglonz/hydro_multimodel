import logging

from dMC.dataset_modules import BaseDataset
from dMC.dataset_modules.utils import (
    create_hydrofabric_attributes,
    create_hydrofabric_observations,
    scale,
)
from data.utils.Hydrofabric import Hydrofabric
from dMC.dataset_modules.utils.Mapping import MeritMap
from dMC.dataset_modules.utils.Network import Network

log = logging.getLogger(__name__)


class TemporallyBatchedDataset(BaseDataset):
    def __init__(self, **kwargs) -> None:
        self.attributes = kwargs["attributes"]
        self.attribute_statistics = kwargs["attribute_statistics"]
        self.cfg = kwargs["cfg"]
        self.dates = kwargs["dates"]
        self.dropout = kwargs["dropout"]
        self.global_to_zone_mapping = kwargs["global_to_zone_mapping"]
        self.gage_dict = kwargs["gage_dict"]
        self.observations = kwargs["observations"]
        self.zone_to_global_mapping = kwargs["zone_to_global_mapping"]

        data = kwargs["data"]
        network = Network(
            attributes=self.attributes,
            cfg=self.cfg,
            data=data,
            dropout=self.dropout,
            gage_dict=self.gage_dict,
            global_to_zone_mapping=self.global_to_zone_mapping,
            zone_to_global_mapping=self.zone_to_global_mapping,
        )
        mapping = MeritMap(cfg=self.cfg, dates=self.dates, network=network)
        hydrofabric_attributes = create_hydrofabric_attributes(
            cfg=self.cfg,
            attributes=self.attributes,
            network=network,
            names=[
                "len",
                "len_dir",
                "sinuosity",
                "slope",
                "stream_drop",
                "uparea",
            ],
        )
        normalized_hydrofabric_attributes = scale(
            df=self.attribute_statistics,
            x=hydrofabric_attributes,
            names=[
                "len",
                "len_dir",
                "sinuosity",
                "slope",
                "stream_drop",
                "uparea",
            ],
        )
        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_dict=self.gage_dict,
            network=network,
            observations=self.observations,
        )
        self.hydrofabric = Hydrofabric(
            attributes=hydrofabric_attributes,
            dates=self.dates,
            mapping=mapping,
            network=network,
            normalized_attributes=normalized_hydrofabric_attributes,
            observations=hydrofabric_observations,
        )

    def __len__(self):
        """
        Returns the total time interval that we're sampling from
        """
        return len(self.dates.daily_time_range)

    def __getitem__(self, idx) -> int:
        """
        Returns a single idx
        :param idx: the index
        :return:
        """
        return idx

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        indices = args[0]
        if 0 not in indices:
            # interpolation requires the previous day's value in order to correctly run
            prev_day = indices[0] - 1
            indices.insert(0, prev_day)

        self.dates.set_date_range(indices)
        return self.hydrofabric
