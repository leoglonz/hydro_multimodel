import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from core.utils.time import trange_to_array

log = logging.getLogger(__name__)



class BaseDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, *args, **kwargs): #-> 'Hydrofabric'
        """
        Collate function with a flexible signature to allow for different inputs
        in subclasses. Implement this method in subclasses to handle specific
        data collation logic.
        """
        raise NotImplementedError


def random_index(ngrid: int, nt: int, dim_subset: Tuple[int, int],
                 warm_up: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, size=batch_size)
    i_t = np.random.randint(warm_up, nt - rho, size=batch_size)  # or 0+warm_up?
    return i_grid, i_t


def n_iter_nt_ngrid(t_range: Tuple[int, int], config: Dict,
                    x: np.ndarray) -> Tuple[int, int, int, int]:
    nt, ngrid, _ = x.shape
    t = trange_to_array(t_range)
    rho = min(t.shape[0], config['rho'])
    n_iter_ep = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - config['batch_size'] * rho / ngrid
                     / (nt - config['warm_up']))
        )
    )
    return ngrid, n_iter_ep, nt, config['batch_size']


def select_subset(config: Dict,
                  x: np.ndarray,
                  i_grid: np.ndarray,
                  i_t: np.ndarray,
                  rho: int,
                  c: Optional[np.ndarray] = None,
                  tuple_out: bool = False,
                  has_grad: bool = False,
                  warm_up: int = 0
                  ) -> torch.Tensor:
    """
    Select a subset of input array.
    """
    nx = x.shape[-1]
    batch_size = i_grid.shape[0]

    if i_t is not None:
        x_tensor = torch.zeros([rho + warm_up, batch_size, nx], requires_grad=has_grad)
        for k in range(batch_size):
            temp = x[np.arange(i_t[k] - warm_up, i_t[k] + rho), i_grid[k]:i_grid[k] + 1, :]
            x_tensor[:, k:k + 1, :] = torch.from_numpy(temp)
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel (x = Ngrid * Ntime).
            x_tensor = torch.from_numpy(x[i_grid, :]).float()
        else:
            # Used for rho equal to the whole length of time series.
            x_tensor = torch.from_numpy(x[:, i_grid, :]).float()
            rho = x_tensor.shape[0]

    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho + warm_up, axis=1)
        c_tensor = torch.from_numpy(temp).float()

        if tuple_out:
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                c_tensor = c_tensor.cuda()
            return x_tensor, c_tensor
        return torch.cat((x_tensor, c_tensor), dim=2)

    return x_tensor.to(config['device']) if torch.cuda.is_available() else x_tensor


def take_sample_train(config: Dict,
                      dataset_dictionary: Dict[str, np.ndarray], 
                      ngrid_train: int,
                      nt: int,
                      batch_size: int
                      ) -> Dict[str, torch.Tensor]:
    """
    Select random sample of data for training batch.
    """
    dim_subset = (batch_size, config['rho'])
    i_grid, i_t = random_index(ngrid_train, nt, dim_subset, warm_up=config['warm_up'])
    dataset_sample = {
        'iGrid': i_grid,
        'inputs_nn_scaled': select_subset(
            config, dataset_dictionary['inputs_nn_scaled'], i_grid, i_t,
            config['rho'], has_grad=False, warm_up=config['warm_up']
        ),
        'c_nn': torch.tensor(dataset_dictionary['c_nn'][i_grid],
                             device=config['device'], dtype=torch.float32),
        'obs': select_subset(config, dataset_dictionary['obs'], i_grid, i_t,
                             config['rho'], warm_up=config['warm_up'])[config['warm_up']:],
        'x_hydro_model': select_subset(config, dataset_dictionary['x_hydro_model'],
                                       i_grid, i_t, config['rho'], warm_up=config['warm_up']),
        'c_hydro_model': torch.tensor(dataset_dictionary['c_hydro_model'][i_grid],
                                       device=config['device'], dtype=torch.float32)
    }
    return dataset_sample


def take_sample_test(config: Dict, dataset_dictionary: Dict[str, torch.Tensor], 
                     i_s: int, i_e: int) -> Dict[str, torch.Tensor]:
    """
    Take sample of data for testing batch.
    """
    dataset_sample = {}
    for key, value in dataset_dictionary.items():
        if value.ndim == 3:
            # TODO: I don't think we actually need this.
            # Remove the warmup period for all except airtemp_memory and hydro inputs.
            if key in ['airT_mem_temp_model', 'x_hydro_model', 'inputs_nn_scaled']:
                warm_up = 0
            else:
                warm_up = config['warm_up']
            dataset_sample[key] = value[warm_up:, i_s:i_e, :].to(config['device'])
        elif value.ndim == 2:
            dataset_sample[key] = value[i_s:i_e, :].to(config['device'])
        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")
    return dataset_sample
