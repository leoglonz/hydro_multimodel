"""
Low-level data manipulation functions are kept here. 
May contain some near-identical functions in the mean time while merging models.
"""
import torch
import numpy as np
from core.data_processing.normalization import transNorm
from core.utils.time import tRange2Array



def scaling(args, x, y, c):
    """
    creates our datasets
    :param set_name:
    :param args:
    :param time1:
    :param x_total_raw:
    :param y_total_raw:
    :return:  x, y, ngrid, nIterEp, nt
    """
    # initcamels(args, x, y)
    # Normalization
    x_total_scaled = transNorm(
        x, args["varT_NN"] + args["varC_NN"], toNorm=True)
    y_scaled = transNorm(y, args["target"], toNorm=True)
    c_scaled = transNorm(c, args["varC_NN"], toNorm=True)

    return x_total_scaled, y_scaled, c_scaled


# def make_tensor(*values, has_grad=False, dtype=torch.float32, device="cuda"):
#     if len(values) > 1:
#         tensor_list = []
#         for value in values:
#             t = torch.tensor(value, requires_grad=has_grad, dtype=dtype,
#                              device=device)
#             tensor_list.append(t)
#     else:
#         for value in values:
#             if type(value) != torch.Tensor:
#                 tensor_list = torch.tensor(
#                     value, requires_grad=has_grad, dtype=dtype, device=device
#                 )
#             else:
#                 tensor_list = value.clone().detach()

#     return tensor_list


# def create_tensor(rho, mini_batch, x, y):
#     """
#     Creates a data tensor of the input variables and incorporates a sliding window of rho
#     :param mini_batch: min batch length
#     :param rho: the seq len
#     :param x: the x data
#     :param y: the y data
#     :return:
#     """
#     j = 0
#     k = rho
#     _sample_data_x = []
#     _sample_data_y = []
#     for i in range(x.shape[0]):
#         _list_x = []
#         _list_y = []
#         while k < x[0].shape[0]:
#             """In the format: [total basins, basin, days, attributes]"""
#             _list_x.append(x[1, j:k, :])
#             _list_y.append(y[1, j:k, 0])
#             j += mini_batch
#             k += mini_batch
#         _sample_data_x.append(_list_x)
#         _sample_data_y.append(_list_y)
#         j = 0
#         k = rho
#     sample_data_x = torch.tensor(_sample_data_x).float()
#     sample_data_y = torch.tensor(_sample_data_y).float()

#     return sample_data_x, sample_data_y


def create_tensor_list(x, y):
    """
    we want to return the:
    x_list = [[[basin_1, num_samples_x, num_attr_x], [basin_1, num_samples_y, num_attr_y]]
        .
        .
        .
        [[basin_20, num_samples_x, num_attr_x], [basin_20, num_samples_y, num_attr_y]]]
    :param data:
    :return:
    """
    tensor_list = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            _var = (torch.tensor(x[i][j][:, :]), y[i, j])
            tensor_list.append(_var)

    return tensor_list


def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   #hack
        iGrid = np.arange(0,len(iGrid))  # hack
    if nt <= rho:
        iT.fill(0)

    batchSize = iGrid.shape[0]
    if iT is not None:
        # batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho+bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if LCopt is True:
            # used for local calibration kernel: FDC, SMAP...
            if len(x.shape) == 2:
                # Used for local calibration kernel as FDC
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # used for LC-SMAP x=Ngrid*Ntime*Nvar
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho+bufftime, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out



def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT