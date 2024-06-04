import logging
from abc import ABC, abstractmethod

import torch
import numpy as np
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


def randomIndex(ngrid, nt, dimSubset, warm_up=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+warm_up, nt - rho, [batchSize])
    return iGrid, iT


def No_iter_nt_ngrid(time_range, args, x):
    nt, ngrid, nx = x.shape
    t = trange_to_array(time_range)
    if t.shape[0] < args['rho']:
        rho = t.shape[0]
    else:
        rho = args['rho']
    nIterEp = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - args['batch_size'] * rho / ngrid / (nt - args['warm_up']))
        )
    )
    return ngrid, nIterEp, nt, args['batch_size']


def selectSubset(args, x, iGrid, iT, rho, *, c=None, tupleOut=False, has_grad=False, warm_up=0):
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
        out = out.to(args['device'])
    return out


def take_sample_train(args, dataset_dictionary, ngrid_train, nt, batchSize):
    dimSubset = [batchSize, args['rho']]
    iGrid, iT = randomIndex(ngrid_train, nt, dimSubset, warm_up=args['warm_up'])
    dataset_dictionary_sample = dict()
    dataset_dictionary_sample['iGrid'] = iGrid
    dataset_dictionary_sample['inputs_nn_scaled'] = selectSubset(args, dataset_dictionary['inputs_nn_scaled'],
                                                                        iGrid, iT, args['rho'], has_grad=False,
                                                                        warm_up=args['warm_up'])
    dataset_dictionary_sample['c_nn'] = torch.tensor(
        dataset_dictionary['c_nn'][iGrid], device=args['device'], dtype=torch.float32
    )
    # collecting observation samples
    dataset_dictionary_sample['obs'] = selectSubset(
        args, dataset_dictionary['obs'], iGrid, iT, args['rho'], has_grad=False, warm_up=args['warm_up']
    )[args['warm_up']:, :, :]
    # dataset_dictionary_sample['obs'] = converting_flow_from_ft3_per_sec_to_mm_per_day(args,
    #                                                                                          dataset_dictionary_sample[
    #                                                                                              'c_nn'],
    #                                                                                          obs_sample_v)
    # Hydro model sampling
    
    dataset_dictionary_sample['x_hydro_model'] = selectSubset(
        args, dataset_dictionary['x_hydro_model'], iGrid, iT, args['rho'], has_grad=False, warm_up=args['warm_up']
    )
    dataset_dictionary_sample['c_hydro_model'] = torch.tensor(
        dataset_dictionary['c_hydro_model'][iGrid], device=args['device'], dtype=torch.float32
    )

    return dataset_dictionary_sample


def take_sample_test(args, dataset_dictionary, iS, iE) -> dict:
    dataset_dictionary_sample = dict()
    for key in dataset_dictionary.keys():
        if len(dataset_dictionary[key].shape) == 3:
            # we need to remove the warm up period for all except airT_memory and inputs for temp model.
            if (key == 'airT_mem_temp_model') or (key == 'x_temp_model') or (key == 'x_hydro_model') or (
                    key == 'inputs_nn_scaled'):
                warm_up = 0
            else:
                warm_up = args['warm_up']
            dataset_dictionary_sample[key] = dataset_dictionary[key][warm_up:, iS: iE, :].to(
                args['device'])
        elif len(dataset_dictionary[key].shape) == 2:
            dataset_dictionary_sample[key] = dataset_dictionary[key][iS: iE, :].to(
                args['device'])
    return dataset_dictionary_sample


def breakdown_params(self, params_all):
    params_dict = dict()
    params_hydro_model = params_all[:, :, :self.ny]

    # hydro params
    params_dict['hydro_params_raw'] = torch.sigmoid(
        params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.config['nmul']]).view(
        params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
        self.config['nmul'])
    # routing params
    if self.config['routing_hydro_model'] == True:
        params_dict['conv_params_hydro'] = torch.sigmoid(
            params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.config['nmul']:])
    else:
        params_dict['conv_params_hydro'] = None
    return params_dict