import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import zarr
from dMC.conf.config import Config
from dMC.dataset_modules.utils.Dates import Dates
from dMC.dataset_modules.utils.Network import FullZoneNetwork, Network
from tqdm import tqdm

log = logging.getLogger(__name__)

# Disable prototype warnings and such
warnings.filterwarnings(action="ignore", category=UserWarning)


def get_zone_indices(edge_order, global_to_zone_mapping) -> DefaultDict[str, List[int]]:
    zone_indices = [global_to_zone_mapping[global_idx] for global_idx in edge_order]
    list_zone_ids = defaultdict(list)
    [list_zone_ids[zone].append(idx) for zone, idx in zone_indices]
    return list_zone_ids


class MeritMap:
    def __init__(
        self, cfg: Config, dates: Dates, network: Union[FullZoneNetwork, Network]
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dates = dates
        self.network = network
        self.comid_indices_dict = {}
        self.comid_map: Dict[Tuple[str, np.int16]] = {}
        self.tm = torch.empty(0)
        self.write_mapping()

    def get_indices(
        self, zone: str, zone_idx: List[int]
    ) -> Tuple[
        npt.NDArray[np.int16],
        npt.NDArray[np.int16],
        npt.NDArray[np.int16],
    ]:
        _zone_idx_array = np.array(zone_idx)
        _zone_attributes = self.network.global_attributes[zone]
        _segment_idx = _zone_attributes.segment_sorting_index[zone_idx]
        _sorted_indices = np.argsort(_zone_idx_array)
        _sorted_zone_idx = _zone_idx_array[_sorted_indices]
        edge_indices = np.zeros_like(_sorted_indices)
        edge_indices[_sorted_indices] = np.arange(len(_sorted_indices))
        comid_indices = np.sort(np.unique(_segment_idx))

        return _sorted_zone_idx, comid_indices, edge_indices

    def read_data(
        self,
        zone: str,
        sorted_idx: npt.NDArray[np.int16],
        comid_indices: npt.NDArray[np.int16],
        edge_indices: npt.NDArray[np.int16],
    ) -> torch.Tensor:
        try:
            zarr_group = zarr.open_group(
                Path(f"{self.cfg.data_sources.MERIT_TM}/MERIT_FLOWLINES_{zone}"),
                mode="r",
            )
        except FileNotFoundError:
            msg = f"Cannot find the MERIT TM {self.cfg.data_sources.MERIT_TM}/MERIT_FLOWLINES_{zone}."
            log.exception(msg=msg)
            raise FileNotFoundError(msg)
        comid_indices_reshaped = comid_indices.reshape(-1, 1)
        merit_to_edge_tm = zarr_group.TM.vindex[comid_indices_reshaped, sorted_idx][
            :, edge_indices
        ]
        return torch.tensor(merit_to_edge_tm, dtype=torch.float64)

    def write_mapping(self) -> None:
        list_zone_ids = get_zone_indices(
            self.network.edge_order, self.network.global_to_zone_mapping
        )
        tms = defaultdict(torch.Tensor)
        for zone, _zone_idx in tqdm(
            list_zone_ids.items(), desc="\rReading MERIT TM", ncols=140, ascii=True
        ):
            _sorted_zone_idx, comid_indices, edge_indices = self.get_indices(
                zone, _zone_idx
            )
            tms[zone] = self.read_data(
                zone, _sorted_zone_idx, comid_indices, edge_indices
            )
            self.comid_indices_dict[zone] = comid_indices

        comid_indices_flattened = [
            (zone, comid_indic)
            for zone, comid_indices in self.comid_indices_dict.items()
            for comid_indic in comid_indices
        ]
        self.comid_map = {
            comid_indices_flattened[i]: i for i in range(len(comid_indices_flattened))
        }

        self.tm = torch.zeros(
            (len(comid_indices_flattened), len(self.network.edge_order)),
            dtype=torch.float64,
        )
        for zone, _zone_idx in tqdm(
            list_zone_ids.items(),
            desc="\rMapping merit tm to matrix",
            ncols=140,
            ascii=True,
        ):
            zone_tuple = [(zone, item) for item in _zone_idx]
            global_idx = [
                self.network.zone_to_global_mapping[tuple_] for tuple_ in zone_tuple
            ]
            subset_idx = np.array(
                [self.network.global_to_subset_mapping[item] for item in global_idx]
            )
            rows = np.array(
                [
                    self.comid_map[(zone, comid)]
                    for comid in self.comid_indices_dict[zone]
                ]
            ).reshape(-1, 1)
            self.tm[rows, subset_idx] = tms[zone]

        self.tm = self.tm.to_sparse_coo()
