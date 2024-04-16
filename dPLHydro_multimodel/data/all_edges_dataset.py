import logging

import torch
import xarray as xr
from data import BaseDataset
from data.utils import create_hydrofabric_attributes, scale
from data.utils.Hydrofabric import Hydrofabric
from data.utils.Mapping import MeritMap
from data.utils.Network import FullZoneNetwork

log = logging.getLogger(__name__)


class AllEdgesDataset(BaseDataset):
    def __init__(self, **kwargs):
        self.attributes = kwargs["attributes"]
        self.attribute_statistics = kwargs["attribute_statistics"]
        self.cfg = kwargs["cfg"]
        self.dates = kwargs["dates"]
        self.dropout = kwargs["dropout"]
        self.global_to_zone_mapping = kwargs["global_to_zone_mapping"]
        self.gage_dict = kwargs["gage_dict"]
        self.observations = kwargs["observations"]
        self.zone_to_global_mapping = kwargs["zone_to_global_mapping"]

        network = FullZoneNetwork(
            attributes=self.attributes,
            cfg=self.cfg,
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
        hydrofabric_obs = self._mock_obs(network=network)
        self.hydrofabric = Hydrofabric(
            attributes=hydrofabric_attributes,
            dates=self.dates,
            mapping=mapping,
            network=network,
            normalized_attributes=normalized_hydrofabric_attributes,
            observations=hydrofabric_obs,
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
        return self.hydrofabric

    def _mock_obs(self, network: FullZoneNetwork) -> xr.Dataset:
        mock_gages = torch.arange(len(network.edge_order))

        # Create the mock dataset
        mock_data = xr.Dataset(
            data_vars={
                "streamflow": (
                    ("gage_id", "time"),
                    torch.zeros(
                        size=[
                            mock_gages.shape[0],
                            len(self.dates.batch_hourly_time_range),
                        ]
                    ),
                ),
            },
            coords={
                "gage_id": mock_gages,
                "time": self.dates.batch_hourly_time_range,
            },
        )

        return mock_data
