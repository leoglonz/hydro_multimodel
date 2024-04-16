import torch
from utils.master import set_globals

# Set global torch device and dtype.
device, dtype = set_globals()



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
