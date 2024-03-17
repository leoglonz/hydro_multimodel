"""Functional interface."""

import os
import numpy as np
import torch
import time
import tqdm


from mm_interface.master import set_globals
from hydroDL.model import crit
from core.data_processing.data_prep import selectSubset, randomIndex
from core.data_processing.data_loading import loadData
from core.data_processing.normalization import transNorm
from core.data_processing.model import (
    take_sample_test,
    converting_flow_from_ft3_per_sec_to_mm_per_day
)


# Set global torch device and dtype.
device, dtype = set_globals()



def trainEnsemble(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               startEpoch=1,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0,
               prcp_loss_factor = 15,
               smooth_loss_factor = 0,
               ):
    """
    inputs:
    x - input
    z - normalized x input
    y - target, or observed values
    c - constant input, attributes
    """

    batchSize, rho = miniBatch
    if type(x) is tuple or type(x) is list:
        x, z = x

    ngrid, nt, nx = x.shape  # ngrid= # basins, nt= # timesteps, nx= # attributes

    if c is not None:
        nx = nx + c.shape[-1]
    if batchSize >= ngrid:
        # Cannot have more batches than there are basins.
        batchSize = ngrid

    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime)))
        )
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(list(model.parameters()))
    model.zero_grad()

    # Save file.
    if saveFolder is not None:
        os.makedirs(saveFolder, exist_ok=True)
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')

    for iEpoch in range(startEpoch, nEpoch + 1):
        lossEp = 0
        loss_prcp_Ep = 0
        loss_sf_Ep = 0
        # loss_smooth_Ep = 0

        t0 = time.time()
        prog_str = "Epoch " + str(iEpoch) + "/" + str(nEpoch)

        for iIter in tqdm(range(0, nIterEp), desc=prog_str, leave=False):
            # training iterations

            iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)

            xTrain = selectSubset(x, iGrid, iT, rho, bufftime=bufftime)
            yTrain = selectSubset(y, iGrid, iT, rho)
            # zTrain = selectSubset(z, iGrid, iT, rho, c=c, bufftime=bufftime)
            zTrain = selectSubset(z, iGrid, iT, rho, bufftime=bufftime)

            # calculate loss and weights `wt`.
            xP, prcp_loss, prcp_weights = model(xTrain, zTrain, prcp_loss_factor)
            yP = torch.sum(xTrain * prcp_weights, dim=2).unsqueeze(2)

            # Consider the buff time for initialization.
            if bufftime > 0:
                yP = yP[bufftime:,:,:]

            # get loss
            if type(lossFun) in [crit.NSELossBatch, crit.NSESqrtLossBatch]:
                loss_sf = lossFun(yP, yTrain, iGrid)
                loss =  loss_sf + prcp_loss
            else:
                loss_sf = lossFun(yP, yTrain)
                loss = loss_sf + prcp_loss

            loss.backward()
            optim.step()
            optim.zero_grad()
            lossEp = lossEp + loss.item()

            try:
                loss_prcp_Ep = loss_prcp_Ep + prcp_loss.item()
            except:
                pass

            loss_sf_Ep = loss_sf_Ep + loss_sf.item()

            # if iIter % 100 == 0:
            #     print('Iter {} of {}: Loss {:.3f}'.format(iIter, nIterEp, loss.item()))

        # print loss
        lossEp = lossEp / nIterEp
        loss_sf_Ep = loss_sf_Ep / nIterEp
        loss_prcp_Ep = loss_prcp_Ep / nIterEp

        logStr = 'Epoch {} Loss {:.3f}, Streamflow Loss {:.3f}, Precipitation Loss {:.3f}, time {:.2f}'.format(
            iEpoch, lossEp, loss_sf_Ep, loss_prcp_Ep,
            time.time() - t0)
        print(logStr)

        # Save model and loss.
        if saveFolder is not None:
            rf.write(logStr + '\n')
            if iEpoch % saveEpoch == 0:
                # save model
                modelFile = os.path.join(saveFolder,
                                         'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)

    if saveFolder is not None:
        rf.close()

    return model



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


def range_bound_loss(params, lb, ub, scale_factor=15):
    """
    Calculate a loss value based on the distance of the parameters from the
    upper and lower bounds of a pre-defined range using a functional approach.
    
    Args:
        params: Parameters tensor or a list of parameter tensors.
        lb: List or tensor of lower bounds for each parameter.
        ub: List or tensor of upper bounds for each parameter.
        scale_factor: Factor to scale the loss.
    """
    lb = torch.tensor(lb, device)
    ub = torch.tensor(ub, device)
    factor = torch.tensor(factor, device)
    
    loss = 0
    for param, lower, upper in zip(params, lb, ub):
        upper_bound_loss = torch.relu(param - upper).mean()
        lower_bound_loss = torch.relu(lower - param).mean()
        loss += (upper_bound_loss + lower_bound_loss) * scale_factor

    return loss


def weighted_avg(x, weights, weights_scaled, dims):
        """
        Get weighted average.
        """
        device = weights.device
        dtype = weights.dtype
        wavg = torch.zeros(dims, dtype=dtype, device=device,requires_grad=True)
        
        for para in range(weights.shape[2]):
            prcp_wavg = prcp_wavg + weights_scaled[:, :, para] * x[:, :, para]

        return prcp_wavg


def t_sum(tensor, ntp, dim):
    """
    Compute sum.
    """
    return torch.sum(tensor[:,:,:ntp], dim=dim)
