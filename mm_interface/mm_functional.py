"""Functional interface."""

import os
import numpy as np
import torch
import time
from tqdm import tqdm


from mm_interface.master import set_globals
from hydroDL.model_new import crit
from core.data_processing.data_prep import selectSubset, randomIndex
from core.data_processing.data_loading import loadData
from core.data_processing.normalization import transNorm
from core.data_processing.model import (
    take_sample_test,
    converting_flow_from_ft3_per_sec_to_mm_per_day
)


# Set global torch device and dtype.
device, dtype = set_globals()



def train_ensemble(model,
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
