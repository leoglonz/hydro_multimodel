import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import zarr
from tqdm import tqdm

log = logging.getLogger(__name__)


def generate_connectivity(
    zone_to_global_mapping: Dict[Tuple[Any, int], int],
    pairs: DefaultDict[Any, List[Any]],
) -> Tuple[torch.Tensor, List[int], Dict[int, int], Dict[int, int]]:
    warnings.filterwarnings(action="ignore", category=UserWarning)

    def pull_edge_order(
        pairs_: List[Union[Tuple[Optional[int], int], Tuple[Optional[int], Any]]],
    ) -> List[int]:
        _idx = [pair[0] for pair in pairs_]
        unique_indices = []
        seen = set()
        for i in _idx:
            if i is None:
                msg = "Index cannot be None"
                log.exception(msg)
                raise TypeError(msg)
            if i not in seen:
                unique_indices.append(int(i))
                seen.add(i)
        return unique_indices

    global_pairs = []
    for zone, zone_pairs in pairs.items():
        for r, c in zone_pairs:
            global_r = zone_to_global_mapping.get((zone, r))
            global_c = (
                zone_to_global_mapping.get((zone, c)) if not np.isnan(c) else None
            )
            if global_c is not None:
                global_pairs.append((global_r, global_c))
            else:
                global_pairs.append((global_r, np.nan))
    edge_order = pull_edge_order(global_pairs)
    global_to_subset_mapping = {id_: idx for idx, id_ in enumerate(edge_order)}
    subset_to_global_mapping = {v: k for k, v in global_to_subset_mapping.items()}
    rows = []
    cols = []

    for ds_id, up_id in tqdm(
        global_pairs, desc="\rCreating Network Matrix", ncols=140, ascii=True
    ):
        if up_id is not None and not np.isnan(up_id):
            ds_idx = global_to_subset_mapping.get(ds_id)
            up_idx = global_to_subset_mapping.get(up_id)
            rows.append(ds_idx)
            cols.append(up_idx)

    rows_tensor = torch.tensor(rows, dtype=torch.int32)
    cols_tensor = torch.tensor(cols, dtype=torch.int32)
    values = torch.ones(len(rows))
    size = (len(edge_order), len(edge_order))
    sparse_coo = torch.sparse_coo_tensor(
        torch.vstack([rows_tensor, cols_tensor]),
        values,
        size,
        dtype=torch.float64,
    )
    sparse_csr = sparse_coo.to_sparse_csr()
    return (
        sparse_csr,
        edge_order,
        global_to_subset_mapping,
        subset_to_global_mapping,
    )


def remove_duplicate_pairs(pairs):
    unique_pairs = set()
    filtered_pairs = []
    for pair in pairs:
        pair_tuple = tuple(pair)
        if pair_tuple not in unique_pairs:
            unique_pairs.add(pair_tuple)
            filtered_pairs.append(pair)
    return filtered_pairs


class FullZoneNetwork:
    def __init__(self, *args, **kwargs):
        self.global_attributes = kwargs["attributes"]
        self.cfg = kwargs["cfg"]
        self.global_to_zone_mapping = kwargs["global_to_zone_mapping"]
        self.zone_to_global_mapping = kwargs["zone_to_global_mapping"]
        if self.cfg.world_size != len(self.cfg.simulation.zone):
            raise ValueError(
                "The number of zones must be equal to the number of processes"
            )
        self.zone = str(self.cfg.simulation.zone[self.cfg.local_rank])

        pairs = self._get_full_network()
        (
            self.matrix,
            self.edge_order,
            self.global_to_subset_mapping,
            self.subset_to_global_mapping,
        ) = generate_connectivity(self.zone_to_global_mapping, pairs)

        self.gage_information = self._set_plotting_information()

    def _get_full_network(self) -> DefaultDict[Any, List[Any]]:
        gage_coo_root = zarr.open_group(
            Path(self.cfg.data_sources.gage_coo_indices) / self.zone, mode="r"
        )
        # gage_order = defaultdict(list)
        pairs = defaultdict(list)
        zone_uparea = self.global_attributes[self.zone].uparea[:]
        if "full_zone" in gage_coo_root:
            coo = gage_coo_root.full_zone
            sorted_indices = np.argsort(zone_uparea[coo.pairs[:, 0].astype(int)])
            sorted_pairs = coo.pairs[sorted_indices]
            filtered_zone_pairs = remove_duplicate_pairs(sorted_pairs)
            pairs[self.zone] = filtered_zone_pairs
            return pairs
        else:
            raise KeyError(
                "Cannot find full_zone connectivity your data. \n"
                "Please use Marquette and download"
                "the sparse data and set 'run_whole_zone' to True"
            )

    def _set_plotting_information(self) -> Dict[str, np.ndarray]:
        plotting_information = {}
        zone_edge_idx = np.array(
            [
                self.global_to_zone_mapping[key][1]
                for key in self.edge_order
                if key in self.global_to_zone_mapping
            ]
        )
        subset_idx = np.array(
            [self.global_to_subset_mapping[idx] for idx in self.edge_order]
        )
        plotting_information["zone_edge_idx"] = zone_edge_idx
        plotting_information["uparea"] = self.global_attributes[self.zone].uparea
        plotting_information["coords"] = self.global_attributes[self.zone].coords
        plotting_information["crs"] = self.global_attributes[self.zone].crs[0]
        plotting_information["gage_subset_idx"] = subset_idx
        return plotting_information


class Network:
    """
    gage_information: Dict[str, List[int]]
    edge_order: List[int]
    global_to_subset_mapping: Dict[int, int]
    subset_to_global_mapping: Dict[int, int]
    matrix: torch.Tensor
    """

    def __init__(self, *args, **kwargs):
        self.global_attributes = kwargs["attributes"]
        self.cfg = kwargs["cfg"]
        data = kwargs["data"]
        self.dropout = kwargs["dropout"]
        self.gage_dict = kwargs["gage_dict"]
        self.global_to_zone_mapping = kwargs["global_to_zone_mapping"]
        self.zone_to_global_mapping = kwargs["zone_to_global_mapping"]

        pairs, filtered_data = self.find_sparse_network_connectivity(data)
        (
            self.matrix,
            self.edge_order,
            self.global_to_subset_mapping,
            self.subset_to_global_mapping,
        ) = generate_connectivity(self.zone_to_global_mapping, pairs)
        self.gage_information = self.generate_gage_information(
            self.global_to_subset_mapping, filtered_data
        )

    def find_sparse_network_connectivity(
        self, data: List[Tuple[int, str, str]]
    ) -> Tuple[DefaultDict[Any, List[Any]], List[Tuple[int, str, str]]]:
        zone_to_gage_ids = defaultdict(list)
        for index, gid, zone in data:
            zone_to_gage_ids[zone].append((index, gid))
        gage_coo_root = zarr.open_group(
            Path(self.cfg.data_sources.gage_coo_indices), mode="a"
        )
        gage_order = defaultdict(list)
        pairs = defaultdict(list)
        for zone, gage_data in tqdm(
            zone_to_gage_ids.items(),
            desc="\rPulling river graphs from zone files",
            ncols=140,
            ascii=True,
        ):
            zone_uparea = self.global_attributes[zone].uparea[:]
            zone_gage_coo = gage_coo_root.require_group(zone)
            zone_pairs = []
            # zone_old_pairs = []
            for idx, gage_id in gage_data:
                gage_order[zone].append(idx)
                if gage_id in zone_gage_coo:
                    coo = zone_gage_coo[gage_id]
                    sorted_indices = np.argsort(
                        zone_uparea[coo.pairs[:, 0].astype(int)]
                    )
                    sorted_pairs = coo.pairs[
                        sorted_indices
                    ]  # sorting each network by upstream ids
                    zone_pairs.extend(sorted_pairs)
                    # zone_old_pairs.append(sorted_pairs)
                else:
                    raise KeyError(
                        f"Cannot find your gage in the list. \n"
                        f"Please use Marquette and "
                        f"download the sparse data for {gage_id}"
                    )
            filtered_zone_pairs = remove_duplicate_pairs(zone_pairs)
            pairs[zone] = filtered_zone_pairs

        if self.dropout.funcs:
            # Apply dropout
            pairs, gage_order = self.dropout(pairs, gage_order)

            filtered_data = [item for item in data if item[0] in gage_order[item[2]]]
            return pairs, filtered_data
        else:
            return pairs, data

    def generate_gage_information(
        self,
        global_to_subset_mapping: Dict[int, int],
        data: List[Tuple[int, str, str]],
    ) -> DefaultDict[str, List[int]]:
        gage_information = defaultdict(list)

        for gage_idx, _, zone in data:
            try:
                global_idx = self.zone_to_global_mapping[
                    (zone, self.gage_dict["zone_edge_id"][gage_idx])
                ]
            except KeyError as e:
                msg = "Cannot find the global index for the gage"
                log.exception(msg)
                raise KeyError(msg) from e

            try:
                subset_idx = global_to_subset_mapping[global_idx]
            except KeyError as e:
                msg = "Cannot find the subset index for the gage"
                log.exception(msg)
                raise KeyError(msg) from e

            gage_information["gage_dict_idx"].append(gage_idx)
            gage_information["gage_global_idx"].append(global_idx)
            gage_information["gage_subset_idx"].append(subset_idx)

        return gage_information
