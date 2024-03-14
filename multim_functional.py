"""Functional interface."""

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

import random
import tqdm
import torch

from core.data_processing.data_loading import loadData
from core.data_processing.normalization import transNorm
from core.data_processing.model import (
    take_sample_test,
    converting_flow_from_ft3_per_sec_to_mm_per_day
)



def randomseed_config(seed):
    """
    Fix random seeds and set torch functions to deterministic forms for model
    reproducibility.

    seed = None: generate random seed and enable use of non-deterministic fns.
    """
    if seed == None:  # args['randomseed'] is None:
        # generate random seed
        randomseed = int(np.random.uniform(low=0, high=1e6))
        print("random seed updated!")
    else:
        print("Setting seed 0.")
        # randomseed = args['randomseed']
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def test_differentiable_model(args, diff_model):
    """
    This function collects and outputs the model predictions and the corresponding
    observations needed to run statistical analyses.

    If rerunning testing in a Jupyter environment, you will need to re-import args
    as `batch_size` is overwritten in this function and will throw an error if the
    overwrite is attempted a second time.
    """
    warm_up = args["warm_up"]
    nmul = args["nmul"]
    diff_model.eval()
    # read data for test time range
    dataset_dictionary = loadData(args, trange=args["t_test"])
    np.save(os.path.join(args["out_dir"], "x.npy"), dataset_dictionary["x_NN"])  # saves with the overlap in the beginning
    # normalizing
    x_NN_scaled = transNorm(args, dataset_dictionary["x_NN"], varLst=args["varT_NN"], toNorm=True)
    c_NN_scaled = transNorm(args, dataset_dictionary["c_NN"], varLst=args["varC_NN"], toNorm=True)
    c_NN_scaled = np.repeat(np.expand_dims(c_NN_scaled, 0), x_NN_scaled.shape[0], axis=0)
    dataset_dictionary["inputs_NN_scaled"] = np.concatenate((x_NN_scaled, c_NN_scaled), axis=2)
    del x_NN_scaled, dataset_dictionary["x_NN"]
    # converting the numpy arrays to torch tensors:
    for key in dataset_dictionary.keys():
        dataset_dictionary[key] = torch.from_numpy(dataset_dictionary[key]).float()

    # args_mod = args.copy()
    args["batch_size"] = args["no_basins"]
    nt, ngrid, nx = dataset_dictionary["inputs_NN_scaled"].shape

    # Making lists of the start and end indices of the basins for each batch.
    batch_size = args["batch_size"]
    iS = np.arange(0, ngrid, batch_size)    # Start index list.
    iE = np.append(iS[1:], ngrid)   # End.

    list_out_diff_model = []
    for i in tqdm(range(0, len(iS)), unit='Batch'):
        dataset_dictionary_sample = take_sample_test(args, dataset_dictionary, iS[i], iE[i])

        out_diff_model = diff_model(dataset_dictionary_sample)
        # Convert all tensors in the dictionary to CPU
        out_diff_model_cpu = {key: tensor.cpu().detach() for key, tensor in out_diff_model.items()}
        # out_diff_model_cpu = tuple(outs.cpu().detach() for outs in out_diff_model)
        list_out_diff_model.append(out_diff_model_cpu)

    # getting rid of warm-up period in observation dataset and making the dimension similar to
    # converting numpy to tensor
    # y_obs = torch.tensor(np.swapaxes(y_obs[:, warm_up:, :], 0, 1), dtype=torch.float32)
    # c_hydro_model = torch.tensor(c_hydro_model, dtype=torch.float32)
    y_obs = converting_flow_from_ft3_per_sec_to_mm_per_day(args,
                                                           dataset_dictionary["c_NN"],
                                                           dataset_dictionary["obs"][warm_up:, :, :])

    return list_out_diff_model, y_obs


def range_bound_loss(params, lb, ub, scale_factor=15, device='cpu'):
    """
    Calculate a loss value based on the distance of the parameters from the
    upper and lower bounds of a pre-defined range using a functional approach.
    
    Args:
        params: Parameters tensor or a list of parameter tensors.
        lb: List or tensor of lower bounds for each parameter.
        ub: List or tensor of upper bounds for each parameter.
        scale_factor: Factor to scale the loss.
        device: The device where tensors will be stored.
    """
    lb = torch.tensor(lb, device=device)
    ub = torch.tensor(ub, device=device)
    factor = torch.tensor(factor, device=device)
    
    loss = 0
    for param, lower, upper in zip(params, lb, ub):
        upper_bound_loss = torch.relu(param - upper).mean()
        lower_bound_loss = torch.relu(lower - param).mean()
        loss += (upper_bound_loss + lower_bound_loss) * scale_factor

    return loss


def weighted_avg(x, weights, weights_scaled, dims, dtype=torch.float32):
        """
        Get weighted average.
        """
        device = weights.device
        dtype = weights.dtype
        wavg = torch.zeros(dims,requires_grad=True,dtype=dtype).to(device)
        
        for para in range(weights.shape[2]):
            prcp_wavg = prcp_wavg + weights_scaled[:, :, para] * x[:, :, para]

        return prcp_wavg


def t_sum(tensor, ntp, dim):
    """
    Compute sum.
    """
    return torch.sum(tensor[:,:,:ntp], dim=dim)
