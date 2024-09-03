import torch
import numpy as np

class NSELossBatch(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, eps=0.1):
        super(NSELossBatch, self).__init__()
        self.std = stdarray
        self.eps = eps

    def forward(self, output, target, igrid):
        nt = target.shape[0]
        stdse = np.tile(self.std[igrid].T, (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()
        p0 = output[:, :, 0]   # dim: Time*Gage
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = (p - t)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return loss
