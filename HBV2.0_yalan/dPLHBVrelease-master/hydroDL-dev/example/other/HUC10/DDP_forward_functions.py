import torch
import numpy as np
import os

def testModel_multiGPU(model, x, c, device,*, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x) is tuple or type(x) is list:
        x, z = x
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    ny = 5

    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.to(device)

    model.train(mode=False)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct
    # yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)

    # deal with file name to save
    if filePathLst is None:
        filePathLst = ['out' + str(x) for x in range(ny)]
    fLst = list()
    for filePath in filePathLst:
        if os.path.exists(filePath):
            os.remove(filePath)
        f = open(filePath, 'a')
        fLst.append(f)

    # forward for each batch
    for i in range(0, len(iS)):
        print('batch {}'.format(i))
        xTemp = x[iS[i]:iE[i], :, :]
        if c is not None:
            cTemp = np.repeat(
                np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), nt, axis=1)
            xTest = torch.from_numpy(
                np.swapaxes(np.concatenate([xTemp, cTemp], 2), 1, 0)).float()
        else:
            xTest = torch.from_numpy(
                np.swapaxes(xTemp, 1, 0)).float()
        if torch.cuda.is_available():
            xTest = xTest.to(device)
        if z is not None:
            zTemp = z[iS[i]:iE[i], :, :]
            zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
            if torch.cuda.is_available():
                zTest = zTest.to(device)

        yP = model(xTest, zTest)


        # CP-- marks the beginning of problematic merge
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)


        # save output
        for k in range(ny):
            f = fLst[k]
            np.save(f.buffer.name,yOut[:, :, k])
        model.zero_grad()
        torch.cuda.empty_cache()

    for f in fLst:
        f.close()

    return yOut

