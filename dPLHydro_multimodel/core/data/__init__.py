import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from core.utils.time import trange_to_array

log = logging.getLogger(__name__)



class BaseDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, *args, **kwargs): #-> Hydrofabric:
        """
        Collate function with a flexible signature to allow for different inputs
        in subclasses. Implement this method in subclasses to handle specific
        data collation logic.
        """
        raise NotImplementedError


def randomIndex(ngrid, nt, dimSubset, warm_up=0) -> list:
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+warm_up, nt - rho, [batchSize])
    return iGrid, iT


def no_iter_nt_ngrid(time_range, config, x) -> list:
    nt, ngrid, nx = x.shape
    t = trange_to_array(time_range)
    if t.shape[0] < config['rho']:
        rho = t.shape[0]
    else:
        rho = config['rho']
    nIterEp = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - config['batch_size'] * rho / ngrid / (nt - config['warm_up']))
        )
    )
    return ngrid, nIterEp, nt, config['batch_size']


def selectSubset(config, x, iGrid, iT, rho, *, c=None, tupleOut=False, has_grad=False, warm_up=0) -> torch.Tensor:
    nx = x.shape[-1]
    nt = x.shape[0]
    # if x.shape[0] == len(iGrid):   #hack
    #     iGrid = np.arange(0,len(iGrid))  # hack
    #     if nt <= rho:
    #         iT.fill(0)

    if iT is not None:
        batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho + warm_up, batchSize, nx], requires_grad=has_grad)
        for k in range(batchSize):
            temp = x[np.arange(iT[k] - warm_up, iT[k] + rho), iGrid[k] : iGrid[k] + 1, :]
            xTensor[:, k : k + 1, :] = torch.from_numpy(temp)
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel
            # x = Ngrid * Ntime
            xTensor = torch.from_numpy(x[iGrid, :]).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(x[:, iGrid, :]).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho + warm_up, axis=1)
        cTensor = torch.from_numpy(temp).float()

        if tupleOut:
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        # out = out.cuda()
        out = out.to(config['device'])
    return out


def take_sample_train(config, dataset_dictionary, ngrid_train, nt, batchSize) -> dict:
    dimSubset = [batchSize, config['rho']]
    iGrid, iT = randomIndex(ngrid_train, nt, dimSubset, warm_up=config['warm_up'])
    dataset_dictionary_sample = dict()
    dataset_dictionary_sample['iGrid'] = iGrid
    dataset_dictionary_sample['inputs_nn_scaled'] = selectSubset(config, dataset_dictionary['inputs_nn_scaled'],
                                                                        iGrid, iT, config['rho'], has_grad=False,
                                                                        warm_up=config['warm_up'])
    dataset_dictionary_sample['c_nn'] = torch.tensor(
        dataset_dictionary['c_nn'][iGrid], device=config['device'], dtype=torch.float32
    )
    # collecting observation samples
    dataset_dictionary_sample['obs'] = selectSubset(
        config, dataset_dictionary['obs'], iGrid, iT, config['rho'], has_grad=False, warm_up=config['warm_up']
    )[config['warm_up']:, :, :]
    # dataset_dictionary_sample['obs'] = converting_flow_from_ft3_per_sec_to_mm_per_day(config,
    #                                                                                          dataset_dictionary_sample[
    #                                                                                              'c_nn'],
    #                                                                                          obs_sample_v)
    # Hydro model sampling
    
    dataset_dictionary_sample['x_hydro_model'] = selectSubset(
        config, dataset_dictionary['x_hydro_model'], iGrid, iT, config['rho'], has_grad=False, warm_up=config['warm_up']
    )
    dataset_dictionary_sample['c_hydro_model'] = torch.tensor(
        dataset_dictionary['c_hydro_model'][iGrid], device=config['device'], dtype=torch.float32
    )

    return dataset_dictionary_sample


def take_sample_test(config, dataset_dictionary, iS, iE) -> dict:
    dataset_dictionary_sample = dict()
    for key in dataset_dictionary.keys():
        if len(dataset_dictionary[key].shape) == 3:
            # we need to remove the warm up period for all except airT_memory and inputs for temp model.
            if (key == 'airT_mem_temp_model') or (key == 'x_temp_model') or (key == 'x_hydro_model') or (
                    key == 'inputs_nn_scaled'):
                warm_up = 0
            else:
                warm_up = config['warm_up']
            dataset_dictionary_sample[key] = dataset_dictionary[key][warm_up:, iS: iE, :].to(
                config['device'])
        elif len(dataset_dictionary[key].shape) == 2:
            dataset_dictionary_sample[key] = dataset_dictionary[key][iS: iE, :].to(
                config['device'])
    return dataset_dictionary_sample
