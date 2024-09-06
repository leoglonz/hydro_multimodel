import numpy as np
import torch
import time
import os
import hydroDL
from hydroDL.model import  cnn, crit
from hydroDL.model import rnn as rnn
from hydroDL.data import scale
import pandas as pd
import xarray as xr
import zarr
import itertools
import random
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def trainModel(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0,
               startepoch = 0,
               area_info =None,
               gage_key = None,
               z_waterloss = None,
               merit_idx =None,
               water_loss_info=None,
               maxmeritBatchSize = 100000):
    print("Start training on epoch :", startepoch)
    batchSize, rho = miniBatch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v3,rnn.MultiInv_HBVTDModel_water_loss_v5,rnn.MultiInv_HBVTDModel_water_loss_v6,rnn.MultiInv_HBVTDModel_water_loss_v7,rnn.MultiInv_HBVTDModel_water_loss_v8 ]:
        ngrid = len(gage_key)
    if c is not None:
        nx = nx + c.shape[-1]
    if batchSize >= ngrid:
        # batchsize larger than total grids
        batchSize = ngrid

    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime))))
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    maxNMerit = 0
    nmerit_max = maxmeritBatchSize
    for iEpoch in range(startepoch, nEpoch + 1):
        if saveFolder is not None:
            runFile = os.path.join(saveFolder, 'run.csv')
            rf = open(runFile, 'a+')
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            
            t0_iter = time.time()
            # training iterations
            if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                xTrain = selectSubset(x, iGrid, iT, rho, c=c, bufftime=bufftime)
                # xTrain = rho/time * Batchsize * Ninput_var
                yTrain = selectSubset(y, iGrid, iT, rho)
                # yTrain = rho/time * Batchsize * Ntraget_var
                yP = model(xTrain)[bufftime:, :, :]
            if type(model) in [rnn.CudnnLstmModel_R2P]:
                # yP = rho/time * Batchsize * Ntraget_var
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c, tupleOut=True)
                yTrain = selectSubset(y, iGrid, iT, rho)
                yP, Param_R2P = model(xTrain)
            if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                               rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel, rnn.CudnnInvLstmModel,
                               rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel,rnn.MultiInv_SACSMAModel,rnn.MultiInv_HBVTDModel_water_loss,rnn.MultiInv_HBVTDModel_water_loss_v2,rnn.MultiInv_HBVTDModel_water_loss_v4]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                if type(model) in [rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel,rnn.MultiInv_SACSMAModel,rnn.MultiInv_HBVTDModel_water_loss,rnn.MultiInv_HBVTDModel_water_loss_v2,rnn.MultiInv_HBVTDModel_water_loss_v4]:
                    xTrain = selectSubset(x, iGrid, iT, rho, bufftime=bufftime)
                else:
                    xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                yTrain = selectSubset(y, iGrid, iT, rho)
                if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=True)
                elif type(model) in [rnn.CudnnInvLstmModel]: # For smap inv LSTM, HBV Inv
                    # zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=False)
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=False, c=c) # Add the attributes to inv
                elif type(model) in [rnn.MultiInv_HBVModel]:
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c)
                elif type(model) in [rnn.MultiInv_HBVTDModel,rnn.MultiInv_SACSMAModel,rnn.MultiInv_HBVTDModel_water_loss,rnn.MultiInv_HBVTDModel_water_loss_v2 ,rnn.MultiInv_HBVTDModel_water_loss_v4]:
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c, bufftime=bufftime)
                    
                else:
                    zTrain = selectSubset(z, iGrid, iT, rho)
            #     if type(model) in [rnn.MultiInv_HBVTDModel_water_loss ]:

            #         gage_key_batch = np.array(gage_key)[iGrid]
            #         Ai_batch = []
            #         Ac_batch = []
            #         id_list = []
            #         start_id = 0
            #         for gage in gage_key_batch:
            #             unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
            #             uparea = area_info[gage]['uparea']
            #             Ai_batch.extend(unitarea)
            #             Ac_batch.extend(uparea)
            #             id_list.append(range(start_id, start_id+len(unitarea)))
            #             start_id = start_id+len(unitarea)

            #         idx_matric = np.zeros((len(Ai_batch),batchSize))
            #         for ii in range(batchSize):
            #             idx_matric[np.array(id_list[ii]),ii] = 1

            #         yP = model(xTrain, zTrain, Ai_batch, Ac_batch, idx_matric)
                
            #     elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v2 ]:    
            #         gage_key_batch = np.array(gage_key)[iGrid]
            #         Ai_batch = []
            #         Ac_batch = []
            #         id_list = []
            #         start_id = 0
            #         for gage in gage_key_batch:
            #             unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
            #             uparea = area_info[gage]['uparea']
            #             Ai_batch.extend(unitarea)
            #             Ac_batch.extend(uparea)
            #             id_list.append(range(start_id, start_id+len(unitarea)))
            #             start_id = start_id+len(unitarea)

            #         # if iIter % 1 == 0:    
            #         #     print(f"Number of gage in iteration {iIter} is ", len(iGrid))
            #         #     print(f"Number of merit in iteration {iIter} is ", start_id)

            #         idx_matric = np.zeros((len(Ai_batch),batchSize))
            #         for ii in range(batchSize):
            #             idx_matric[np.array(id_list[ii]),ii] = 1
                    

            #         forcing_norm2,attribute_norm2 = z_waterloss
                    
            #         xTrain2 = np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
            #         attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)
                    
            #         for gageidx , gage in enumerate(gage_key_batch):
            #             xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
            #             attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]
                    
                    
            #         attr2 = np.expand_dims(attr2, axis=1)
            #         attr2 = np.repeat(attr2, xTrain2.shape[1], axis=1)
            #         zTrain2 = np.concatenate([xTrain2, attr2], 2) 
                    
            #         zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0,1)).to(zTrain)
            #         yP = model(xTrain, zTrain,zTrain2_torch, Ai_batch, Ac_batch, idx_matric)     

            #     elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v4 ]:    
            #         gage_key_batch = np.array(gage_key)[iGrid]
            #         Ai_batch = []
            #         Ac_batch = []
            #         id_list = []
            #         start_id = 0
            #         for gage in gage_key_batch:
            #             unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
            #             uparea = area_info[gage]['uparea']
            #             Ai_batch.extend(unitarea)
            #             Ac_batch.extend(uparea)
            #             id_list.append(range(start_id, start_id+len(unitarea)))
            #             start_id = start_id+len(unitarea)

            #         # if iIter % 1 == 0:    
            #         #     print(f"Number of gage in iteration {iIter} is ", len(iGrid))
            #         #     print(f"Number of merit in iteration {iIter} is ", start_id)

            #         idx_matric = np.zeros((len(Ai_batch),batchSize))
            #         for ii in range(batchSize):
            #             idx_matric[np.array(id_list[ii]),ii] = 1

            #         _,attribute_norm2 = z_waterloss
                    
            #         attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)
                    
            #         for gageidx , gage in enumerate(gage_key_batch):
            #             attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]

                    
            #         zTrain2_torch = torch.from_numpy(attr2).to(zTrain)
            #         yP = model(xTrain, zTrain,zTrain2_torch, Ai_batch, Ac_batch, idx_matric)   


            #     else:
            #         yP = model(xTrain, zTrain)

            # if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v3]:     
            #     iGrid0, iT0 = randomIndex(len(gage_key), nt, [batchSize, rho], bufftime=bufftime) 
            #     nmerit = water_loss_info[0]
                
                
            #     iGrid = reduce_sum_until_target(nmerit,np.array(iGrid0), target_sum=3000,batchsize = batchSize)
            #     iT = iT0[:len(iGrid)]

            #     yTrain = selectSubset(y, iGrid, iT, rho)
            #     gage_key_batch = np.array(gage_key)[iGrid]
            #     Ai_batch = []
            #     Ac_batch = []
            #     id_list = []
            #     start_id = 0
            #     for gage in gage_key_batch:
            #         unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
            #         uparea = area_info[gage]['uparea']
            #         Ai_batch.extend(unitarea)
            #         Ac_batch.extend(uparea)
            #         id_list.append(range(start_id, start_id+len(unitarea)))
            #         start_id = start_id+len(unitarea)
                

            #     idx_matric = np.zeros((len(Ai_batch),len(iGrid)))
            #     for ii in range(len(iGrid)):
            #         idx_matric[np.array(id_list[ii]),ii] = 1
                
            #     forcing_norm2,attribute_norm2 = z_waterloss

            #     xTrain = np.full((len(Ac_batch),rho+bufftime,x.shape[-1]),np.nan)

            #     xTrain2 = np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
            #     attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)


            #     for gageidx , gage in enumerate(gage_key_batch):
            #         xTrain[np.array(id_list[gageidx]),:,:] = x[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
            #         xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
            #         attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]
                
                
            #     attr2 = np.expand_dims(attr2, axis=1)
            #     attr2 = np.repeat(attr2, xTrain2.shape[1], axis=1)
                
            #     zTrain2 = np.concatenate([xTrain2, attr2], 2) 

            #     xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).float().cuda()
            #     zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0,1)).float().cuda()
            #     yP = model(xTrain_torch, zTrain2_torch, Ai_batch, Ac_batch, idx_matric)       
                               
            # if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v5 ]:  
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
            #     yTrain = selectSubset(y, iGrid, iT, rho)  
            #     gage_key_batch = np.array(gage_key)[iGrid]
            #     Ai_batch = []
            #     Ac_batch = []
            #     id_list = []
            #     start_id = 0
            #     for gage in gage_key_batch:
            #         unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
            #         uparea = area_info[gage]['uparea']
            #         Ai_batch.extend(unitarea)
            #         Ac_batch.extend(uparea)
            #         id_list.append(range(start_id, start_id+len(unitarea)))
            #         start_id = start_id+len(unitarea)
            #     idx_matric = np.zeros((len(Ai_batch),batchSize))
            #     for ii in range(batchSize):
            #         idx_matric[np.array(id_list[ii]),ii] = 1
                

            #     forcing_norm2,attribute_norm2 = z_waterloss
            #     xTrain =  np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
            #     xTrain2 = np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
            #     attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)
                
            #     for gageidx , gage in enumerate(gage_key_batch):
            #         xTrain[np.array(id_list[gageidx]),:,:] = x[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
            #         xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
            #         attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]
                
      
            #     zTrain2_torch = torch.from_numpy(np.swapaxes(xTrain2, 0,1)).to(yTrain)
            #     xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).to(yTrain)
            #     attr_torch = torch.from_numpy(attr2).to(yTrain)
            #     yP = model(xTrain_torch, zTrain2_torch,attr_torch, Ai_batch, Ac_batch, idx_matric)  


            # # if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v6]: 
                 
            # #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                 
            # #     gage_key_batch = np.array(gage_key)[iGrid]
            # #     id_list = []
            # #     start_id = 0
                
            # #     for gage_idx, gage in enumerate(gage_key_batch):
            # #         if(start_id+len(merit_idx[gage])) > nmerit_max:
            # #             print("Minibatch will be shrinked to ",gage_idx,"  since it has nmerit of ", start_id+len(merit_idx[gage]))
                        
            # #             iGrid=iGrid[:gage_idx]
            # #             iT = iT[:gage_idx]
                        
            # #             gage_key_batch = gage_key_batch[:gage_idx]
                        
            # #             break

            # #         id_list.append(range(start_id, start_id+len(merit_idx[gage])))
                    
            # #         start_id = start_id+len(merit_idx[gage])
                
            # #     yTrain = selectSubset(y, iGrid, iT, rho)
                

            # #     if(start_id>maxNMerit): maxNMerit = start_id
                
            # #     forcing_norm2,attribute_norm2 = z_waterloss
            # #     xTrain =  np.full((start_id,rho+bufftime,forcing_norm2.shape[-1]),np.nan)
            # #     xTrain2 = np.full((start_id,rho+bufftime,forcing_norm2.shape[-1]),np.nan)
            # #     attr2 = np.full((start_id,attribute_norm2.shape[-1]),np.nan)
                
            # #     idx_matric = np.zeros((start_id,len(gage_key_batch)))

            # #     Merit_all,Ac_all,Ai_all=water_loss_info
            # #     Ai_batch = []
            # #     Ac_batch = []
            # #     for gageidx , gage in enumerate(gage_key_batch):

            # #         idx_matric[np.array(id_list[gageidx]),gageidx] = 1
                    
            # #         Ai_batch.extend(Ai_all[np.array(merit_idx[gage]).astype(int)]/np.array(Ai_all[np.array(merit_idx[gage]).astype(int)]).sum())
            # #         Ac_batch.extend(Ac_all[np.array(merit_idx[gage]).astype(int)])
            # #         xTrain[np.array(id_list[gageidx]),:,:] = x[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
            # #         xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
            # #         attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]   

            # #     xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).to(yTrain)
            # #     attr_torch = torch.from_numpy(attr2).to(yTrain)

            # #     attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTrain2.shape[1], axis=1)
            # #     zTrain2 = np.concatenate((xTrain2,attr2_expand),axis = -1)
            # #     zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0, 1)).to(yTrain)
                
            # #     yP = model(xTrain_torch, zTrain2_torch,attr_torch, Ai_batch, Ac_batch, idx_matric)    
            # #     #yP = yP[bufftime:,:,:]
            
            
            ####################################################################
            if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v6]: 
                 
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                 
                gage_key_batch = np.array(gage_key)[iGrid]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                
                for gage_idx, gage in enumerate(gage_key_batch):
                    if(start_id+len(area_info[gage]['unitarea'])) > nmerit_max:
                        print("Minibatch will be shrinked to ",gage_idx,"  since it has nmerit of ", start_id+len(area_info[gage]['unitarea']))
                        
                        iGrid=iGrid[:gage_idx]
                        iT = iT[:gage_idx]
                        
                        gage_key_batch = gage_key_batch[:gage_idx]
                        
                        break

                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    
                    start_id = start_id+len(unitarea)
                


                yTrain = selectSubset(y, iGrid, iT, rho)
                
                # for ii in range(len(gage_key_batch)):
                #     idx_matric[np.array(id_list[ii]),ii] = 1
                if(len(Ai_batch)>maxNMerit): maxNMerit = len(Ai_batch)
                
                forcing_norm2,attribute_norm2 = z_waterloss
                xTrain =  np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                xTrain2 = np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)
                
                idx_matric = np.zeros((len(Ai_batch),len(gage_key_batch)))
                for gageidx , gage in enumerate(gage_key_batch):

                    idx_matric[np.array(id_list[gageidx]),gageidx] = 1

                    xTrain[np.array(id_list[gageidx]),:,:] = x[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]
                
      
                if np.isnan(xTrain).any():
                    raise Exception("xTrain has nan at Iteration ,", iIter)
                xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).to(yTrain)
                attr_torch = torch.from_numpy(attr2).to(yTrain)

                attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTrain2.shape[1], axis=1)
                zTrain2 = np.concatenate((xTrain2,attr2_expand),axis = -1)
                zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0, 1)).to(yTrain)

                if np.isnan(zTrain2).any():
                    raise Exception("zTrain2 has nan at Iteration ,", iIter)

                if np.isnan(np.array(Ai_batch)).any():
                    raise Exception("Ai_batch has nan at Iteration ,", iIter)

                if np.isnan(np.array(Ac_batch)).any():
                    raise Exception("Ac_batch has nan at Iteration ,", iIter)
                if (iEpoch>= 3 and iIter > 800):
                    yP = model(xTrain_torch, zTrain2_torch,attr_torch, Ai_batch, Ac_batch, idx_matric)    
            
            
            if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v7]:  
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                yTrain = selectSubset(y, iGrid, iT, rho)  
                gage_key_batch = np.array(gage_key)[iGrid]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)
                idx_matric = np.zeros((len(Ai_batch),batchSize))
                for ii in range(batchSize):
                    idx_matric[np.array(id_list[ii]),ii] = 1
                

                forcing_norm2,attribute_norm2 = z_waterloss
                xTrain =  np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                xTrain2 = np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)
                
                for gageidx , gage in enumerate(gage_key_batch):
                    xTrain[np.array(id_list[gageidx]),:,:] = x[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]
                
      

                xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).to(yTrain)

                attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTrain2.shape[1], axis=1)
                zTrain2 = np.concatenate((xTrain2,attr2_expand),axis = -1)
                zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0, 1)).to(yTrain)
                yP = model(xTrain_torch, zTrain2_torch, Ai_batch, Ac_batch, idx_matric)                 



            if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v8]: 
                 
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                 
                gage_key_batch = np.array(gage_key)[iGrid]
                id_list = []
                start_id = 0
                
                for gage_idx, gage in enumerate(gage_key_batch):
                    if(start_id+len(merit_idx[gage])) > nmerit_max:
                        print("Minibatch will be shrinked to ",gage_idx,"  since it has nmerit of ", start_id+len(merit_idx[gage]))
                        
                        iGrid=iGrid[:gage_idx]
                        iT = iT[:gage_idx]
                        
                        gage_key_batch = gage_key_batch[:gage_idx]
                        
                        break

                    id_list.append(range(start_id, start_id+len(merit_idx[gage])))
                    
                    start_id = start_id+len(merit_idx[gage])
                
                yTrain = selectSubset(y, iGrid, iT, rho)
                

                if(start_id>maxNMerit): maxNMerit = start_id
                
                forcing_norm2,attribute_norm2,norm_gage_area = z_waterloss
                xTrain =  np.full((start_id,rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                xTrain2 = np.full((start_id,rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                attr2 = np.full((start_id,attribute_norm2.shape[-1]),np.nan)
                attr_rout = np.full((len(gage_key_batch),attribute_norm2.shape[-1]),np.nan)

                idx_matric = np.zeros((start_id,len(gage_key_batch)))

                Merit_all,Ac_all,Ai_all=water_loss_info
                Ai_batch = []
                Ac_batch = []
                for gageidx , gage in enumerate(gage_key_batch):

                    idx_matric[np.array(id_list[gageidx]),gageidx] = 1
                    
                    Ai_batch.extend(Ai_all[np.array(merit_idx[gage]).astype(int)]/np.array(Ai_all[np.array(merit_idx[gage]).astype(int)]).sum())
                    Ac_batch.extend(Ac_all[np.array(merit_idx[gage]).astype(int)])
                    xTrain[np.array(id_list[gageidx]),:,:] = x[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]   
                    attr_rout[gageidx,:] =np.sum(attribute_norm2[np.array(merit_idx[gage]).astype(int),:] * Ai_all[np.array(merit_idx[gage]).astype(int),np.newaxis]/np.array(Ai_all[np.array(merit_idx[gage]).astype(int)]).sum(), axis = 0) 
                    attr_rout[gageidx,-1] = norm_gage_area[gageidx,0]
                xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).to(yTrain)
                attr_torch = torch.from_numpy(attr2).to(yTrain)
                attr_rout_torch = torch.from_numpy(attr_rout).to(yTrain)
                attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTrain2.shape[1], axis=1)
                zTrain2 = np.concatenate((xTrain2,attr2_expand),axis = -1)
                zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0, 1)).to(yTrain)
                
                yP = model(xTrain_torch, zTrain2_torch,attr_torch,attr_rout_torch, Ai_batch, Ac_batch, idx_matric)   













            if type(model) in [cnn.LstmCnn1d]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                # xTrain = rho/time * Batchsize * Ninput_var
                xTrain = xTrain.permute(1, 2, 0)
                yTrain = selectSubset(y, iGrid, iT, rho)
                # yTrain = rho/time * Batchsize * Ntraget_var
                yTrain = yTrain.permute(1, 2, 0)[:, :, int(rho/2):]
                yP = model(xTrain)
            # if type(model) in [hydroDL.model.rnn.LstmCnnCond]:
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            #     xTrain = selectSubset(x, iGrid, iT, rho)
            #     yTrain = selectSubset(y, iGrid, iT, rho)
            #     zTrain = selectSubset(z, iGrid, None, None)
            #     yP = model(xTrain, zTrain)
            # if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            #     xTrain = selectSubset(x, iGrid, iT, rho)
            #     yTrain = selectSubset(y, iGrid, iT + model.ct, rho - model.ct)
            #     zTrain = selectSubset(z, iGrid, iT, rho)
            #     yP = model(xTrain, zTrain)
            else:
                Exception('unknown model')
            # # consider the buff time for initialization
            # if bufftime > 0:
            #     yP = yP[bufftime:,:,:]
            ## temporary test for NSE loss
            if (iEpoch>= 3 and iIter > 800):
                if type(lossFun) in [crit.NSELossBatch, crit.NSESqrtLossBatch]:
                    loss = lossFun(yP, yTrain, iGrid)
                else:
                    loss = lossFun(yP, yTrain)
                loss.backward()
                optim.step()
                model.zero_grad()
                lossEp = lossEp + loss.item()
            torch.cuda.empty_cache()
            # print(iIter, '  ', loss.item())
            # if iIter == 223:
            #     print('This is the error point')
            #     print('Debug start')

            if iIter % 50 == 0:

                logStr = 'Iter {} of {}: Loss {:.3f} time {:.2f} maximum number of merits {}'.format(iIter, nIterEp, loss.item(),time.time() - t0_iter,maxNMerit)
                
                print(logStr)

                rf.write(logStr + '\n')
        # print loss
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, lossEp,
            time.time() - t0)
        print(logStr)
        # save model and loss
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



def cleanup():
    dist.destroy_process_group()

def trainModel_multiGPU(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0,
               startepoch = 0,
               area_info =None,
               gage_key = None,
               z_waterloss = None,
               merit_idx =None,
               water_loss_info=None,
               GPUlist = None):
    
    randomseedlist = [111111, 222222,333333,444444,555555,666666]


    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    np.random.seed(randomseedlist[rank])
    random.seed(randomseedlist[rank])

    world_size = dist.get_world_size()
    device = torch.device("cuda:" + str(GPUlist[rank]))
    model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[GPUlist[rank]])

    optim = torch.optim.Adadelta(ddp_model.parameters())

    batchSize,rho = miniBatch
    x,z = x

    
    # gage_key = gage_key[0]
    # area_info = area_info[0]
    # merit_idx = merit_idx[0]
    ngrid = len(gage_key)
    nt =  y.shape[1]
    #print('ngrid is ', ngrid)
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime))))

    nIterEp = int(nIterEp/world_size)

    if rank == 0:
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')
    maxNMerit = 0
    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            t0_iter = time.time()
            # training iterations
            if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v6]:  
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                #print("rank ", rank, ' iGrid ', iGrid[:5],' iT ', iT[:5])
                yTrain = selectSubset_multiGPUs(y, iGrid, iT, rho,device = device)
                gage_key_batch = np.array(gage_key)[iGrid]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)
                idx_matric = np.zeros((len(Ai_batch),batchSize))
                for ii in range(batchSize):
                    idx_matric[np.array(id_list[ii]),ii] = 1

                if(len(Ai_batch)>maxNMerit): maxNMerit = len(Ai_batch)

                forcing_norm2,attribute_norm2 = z_waterloss
                xTrain =  np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                xTrain2 = np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)


                for gageidx , gage in enumerate(gage_key_batch):
                    xTrain[np.array(id_list[gageidx]),:,:] = x[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]
                
        

                xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).to(yTrain)
                attr_torch = torch.from_numpy(attr2).to(yTrain)

                attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTrain2.shape[1], axis=1)
                zTrain2 = np.concatenate((xTrain2,attr2_expand),axis = -1)
                zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0, 1)).to(yTrain)
            
                dist.barrier()
                optim.zero_grad()
                yP = ddp_model(xTrain_torch, zTrain2_torch,attr_torch, Ai_batch, Ac_batch, idx_matric) 






            if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v7]:  
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                yTrain = selectSubset_multiGPUs(y, iGrid, iT, rho,device = device)
                gage_key_batch = np.array(gage_key)[iGrid]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)
                idx_matric = np.zeros((len(Ai_batch),batchSize))
                for ii in range(batchSize):
                    idx_matric[np.array(id_list[ii]),ii] = 1
                
                if(len(Ai_batch)>maxNMerit): maxNMerit = len(Ai_batch)

                forcing_norm2,attribute_norm2 = z_waterloss
                xTrain =  np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                xTrain2 = np.full((len(Ac_batch),rho+bufftime,forcing_norm2.shape[-1]),np.nan)
                attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)
                
                for gageidx , gage in enumerate(gage_key_batch):
                    xTrain[np.array(id_list[gageidx]),:,:] = x[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    xTrain2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),iT[gageidx]-bufftime:iT[gageidx]+rho,:]
                    attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]
                
      

                xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).to(yTrain)

                attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTrain2.shape[1], axis=1)
                zTrain2 = np.concatenate((xTrain2,attr2_expand),axis = -1)
                zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0, 1)).to(yTrain)
                dist.barrier()
                optim.zero_grad()
                yP = ddp_model(xTrain_torch, zTrain2_torch, Ai_batch, Ac_batch, idx_matric)                 











            if type(lossFun) in [crit.NSELossBatch, crit.NSESqrtLossBatch]:
                loss = lossFun(yP, yTrain, iGrid)
            else:
                loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()


            # dist.barrier()
            #torch.distributed.all_reduce(loss, op=dist.ReduceOp.AVG)
           # if rank == 0 or rank == 1:
            lossEp = lossEp + loss.item()

            if iIter % 10 == 0:
                print('Iter {} of {}: Loss {:.3f} time {:.2f} maximum number of merits {}'.format(iIter, nIterEp, loss.item(),time.time() - t0_iter,maxNMerit))
        # print loss
        if rank == 0:
            lossEp = lossEp / nIterEp
            logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
                iEpoch, lossEp,
                time.time() - t0)
            print(logStr)

            if saveFolder is not None:
                rf.write(logStr + '\n')
                if iEpoch % saveEpoch == 0:
                    # save model
                    modelFile = os.path.join(saveFolder,
                                             'model_Ep' + str(iEpoch) + '.pt')
                    torch.save(model, modelFile)
    if rank == 0:
        if saveFolder is not None:
            rf.close()
    cleanup()





def trainModel_merit(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0,
               startepoch=0,
               area_info=None,
               gage_key=None,
               z_waterloss=None,
               merit_idx=None,
               water_loss_info=None):
    print("Start training on epoch :", startepoch)
    batchSize, rho = miniBatch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v3, rnn.MultiInv_HBVTDModel_water_loss_v5, rnn.MultiInv_HBVTDModel_water_loss_v6]:
        ngrid = len(gage_key)

    gage_idx = np.arange(0, ngrid)
    time_idx = np.arange(bufftime, nt-rho, rho)
    time_idx  = np.append(time_idx, nt-rho)

    batch_list = list(itertools.product(gage_idx, time_idx))
    random.shuffle(batch_list)

    # Extract the first elements from each tuple in c
    iGrid_list = [elem[0] for elem in batch_list]

    # Extract the second elements from each tuple in c
    iT_list = [elem[1] for elem in batch_list]


    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    minibatchs = np.arange(0, len(batch_list), batchSize)
    minibatchs = np.append(minibatchs, len(batch_list))
    for iEpoch in range(startepoch, nEpoch + 1):
        if saveFolder is not None:
            runFile = os.path.join(saveFolder, 'run.csv')
            rf = open(runFile, 'a+')
        lossEp = 0
        t0 = time.time()
        for iIter in range(len(minibatchs[:-1])):
            t0_iter = time.time()

            # training iterations
            iGrid = np.array(iGrid_list[minibatchs[iIter]:minibatchs[iIter+1]])
            iT = np.array(iT_list[minibatchs[iIter]: minibatchs[iIter + 1]])
            if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v5]:
                yTrain = selectSubset(y, iGrid, iT, rho)
                gage_key_batch = np.array(gage_key)[iGrid]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea'] / np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id + len(unitarea)))
                    start_id = start_id + len(unitarea)

                idx_matric = np.zeros((len(Ai_batch), len(iGrid)))
                for ii in range(len(iGrid)):
                    idx_matric[np.array(id_list[ii]), ii] = 1

                forcing_norm2, attribute_norm2 = z_waterloss
                xTrain = np.full((len(Ac_batch), rho + bufftime, forcing_norm2.shape[-1]), np.nan)
                xTrain2 = np.full((len(Ac_batch), rho + bufftime, forcing_norm2.shape[-1]), np.nan)
                attr2 = np.full((len(Ac_batch), attribute_norm2.shape[-1]), np.nan)

                for gageidx, gage in enumerate(gage_key_batch):
                    xTrain[np.array(id_list[gageidx]), :, :] = x[np.array(merit_idx[gage]).astype(int),
                                                               iT[gageidx] - bufftime:iT[gageidx] + rho, :]
                    xTrain2[np.array(id_list[gageidx]), :, :] = forcing_norm2[np.array(merit_idx[gage]).astype(int),
                                                                iT[gageidx] - bufftime:iT[gageidx] + rho, :]
                    attr2[np.array(id_list[gageidx]), :] = attribute_norm2[np.array(merit_idx[gage]).astype(int), :]

                zTrain2_torch = torch.from_numpy(np.swapaxes(xTrain2, 0, 1)).to(yTrain)
                xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0, 1)).to(yTrain)
                attr_torch = torch.from_numpy(attr2).to(yTrain)
                yP = model(xTrain_torch, zTrain2_torch, attr_torch, Ai_batch, Ac_batch, idx_matric)


            elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v6]:
                yTrain = selectSubset(y, iGrid, iT, rho)
                gage_key_batch = np.array(gage_key)[iGrid]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea'] / np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id + len(unitarea)))
                    start_id = start_id + len(unitarea)
                
                idx_matric = np.zeros((len(Ai_batch), len(iGrid)))
                for ii in range(len(iGrid)):
                    idx_matric[np.array(id_list[ii]), ii] = 1
                
                forcing_norm2, attribute_norm2 = z_waterloss
                xTrain = np.full((len(Ac_batch), rho + bufftime, forcing_norm2.shape[-1]), np.nan)
                xTrain2 = np.full((len(Ac_batch), rho + bufftime, forcing_norm2.shape[-1]), np.nan)
                attr2 = np.full((len(Ac_batch), attribute_norm2.shape[-1]), np.nan)

                for gageidx, gage in enumerate(gage_key_batch):
                    xTrain[np.array(id_list[gageidx]), :, :] = x[np.array(merit_idx[gage]).astype(int),
                                                               iT[gageidx] - bufftime:iT[gageidx] + rho, :]
                    xTrain2[np.array(id_list[gageidx]), :, :] = forcing_norm2[np.array(merit_idx[gage]).astype(int),
                                                                iT[gageidx] - bufftime:iT[gageidx] + rho, :]
                    attr2[np.array(id_list[gageidx]), :] = attribute_norm2[np.array(merit_idx[gage]).astype(int), :]

                
                xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0, 1)).to(yTrain)

                
                attr_torch = torch.from_numpy(attr2).to(yTrain)
                attr2_expand = np.repeat(attr2, xTrain2.shape[1], axis=1)
                zTrain2 = np.concatenate((xTrain2,attr2_expand),axis = -1)
                zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0, 1)).to(yTrain)
                yP = model(xTrain_torch, zTrain2_torch, attr_torch, Ai_batch, Ac_batch, idx_matric)




            else:
                Exception('unknown model')
            # # consider the buff time for initialization
            # if bufftime > 0:
            #     yP = yP[bufftime:,:,:]
            ## temporary test for NSE loss
            if type(lossFun) in [crit.NSELossBatch, crit.NSESqrtLossBatch]:
                loss = lossFun(yP, yTrain, iGrid)
            else:
                loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()

            # print(iIter, '  ', loss.item())
            # if iIter == 223:
            #     print('This is the error point')
            #     print('Debug start')

            if iIter % 50 == 0:
                logStr = 'Iter {} of {}: Loss {:.3f} time {:.2f}'.format(iIter, len(minibatchs[:-1]), loss.item(),
                                                                         time.time() - t0_iter)
                print(logStr)

                rf.write(logStr + '\n')
        # print loss
        lossEp = lossEp / len(minibatchs[:-1])
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, lossEp,
            time.time() - t0)
        print(logStr)
        # save model and loss
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

def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model

def loadModel_multiGPU(outFolder, device, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile, map_location=device)
    return model

def testModel(model, x,z_waterloss, c, *, water_loss_info = None, area_info =None,  gage_key = None, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x) is tuple or type(x) is list:
        x, z = x
        if type(model) is rnn.CudnnLstmModel:
            # For Cudnn, only one input. First concat inputs and obs
            x = np.concatenate([x, z], axis=2)
            z = None
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    if type(model) in [rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel,rnn.MultiInv_HBVTDModel_water_loss,rnn.MultiInv_HBVTDModel_water_loss_v2,rnn.MultiInv_HBVTDModel_water_loss_v4]:
        ny=5 # streamflow
    elif type(model) in [rnn.MultiInv_SACSMAModel]:
        ny=2 # streamflo
    else:
        ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    torch.set_grad_enabled(False)

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
            xTest = xTest.cuda()
        if z is not None:
            if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                if len(z.shape) == 2:
                    # Used for local calibration kernel as FDC
                    # x = Ngrid * Ntime
                    zTest = torch.from_numpy(z[iS[i]:iE[i], :]).float()
                elif len(z.shape) == 3:
                    # used for LC-SMAP x=Ngrid*Ntime*Nvar
                    zTest = torch.from_numpy(np.swapaxes(z[iS[i]:iE[i], :, :], 1, 2)).float()
            else:
                zTemp = z[iS[i]:iE[i], :, :]
                # if type(model) in [rnn.CudnnInvLstmModel]: # Test SMAP Inv with attributes
                #     cInv = np.repeat(
                #         np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), zTemp.shape[1], axis=1)
                #     zTemp = np.concatenate([zTemp, cInv], 2)
                zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
            if torch.cuda.is_available():
                zTest = zTest.cuda()
        if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
            # if z is not None:
            #     xTest = torch.cat((xTest, zTest), dim=2)
            yP = model(xTest)
            if doMC is not False:
                ySS = np.zeros(yP.shape)
                yPnp=yP.detach().cpu().numpy()
                for k in range(doMC):
                    # print(k)
                    yMC = model(xTest, doDropMC=True).detach().cpu().numpy()
                    ySS = ySS+np.square(yMC-yPnp)
                ySS = np.sqrt(ySS)/doMC
        if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                           rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel, rnn.CudnnInvLstmModel,
                           rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel,rnn.MultiInv_SACSMAModel,rnn.MultiInv_HBVTDModel_water_loss,rnn.MultiInv_HBVTDModel_water_loss_v2,rnn.MultiInv_HBVTDModel_water_loss_v4]:

            if type(model) in [rnn.MultiInv_HBVTDModel_water_loss]:

                gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)


                idx_matric = np.zeros((len(Ai_batch),(iE[i]-iS[i])))
                for ii in range(iE[i]-iS[i]):
                    idx_matric[np.array(id_list[ii]),ii] = 1
                
                yP = model(xTest, zTest, Ai_batch, Ac_batch, idx_matric)


            # elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v2 ]:    
            #     gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
            #     Ai_batch = []
            #     Ac_batch = []
            #     id_list = []
            #     start_id = 0
            #     for gage in gage_key_batch:
            #         unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
            #         uparea = area_info[gage]['uparea']
            #         Ai_batch.extend(unitarea)
            #         Ac_batch.extend(uparea)
            #         id_list.append(range(start_id, start_id+len(unitarea)))
            #         start_id = start_id+len(unitarea)


            #     idx_matric = np.zeros((len(Ai_batch),(iE[i]-iS[i])))
            #     for ii in range(batchSize):
            #         idx_matric[np.array(id_list[ii]),ii] = 1
                     

            #     forcing_norm2,attribute_norm2 = z_waterloss
                
            #     xTest2 = np.full((len(Ac_batch),x.shape[1],forcing_norm2.shape[-1]),np.nan)
            #     attr2 = np.full((len(Ac_batch),attribute_norm2.shape[-1]),np.nan)
                
            #     for gageidx , gage in enumerate(gage_key_batch):
            #         xTest2[np.array(id_list[gageidx]),:,:] = forcing_norm2[np.array(merit_idx[gage]).astype(int),:,:]
            #         attr2[np.array(id_list[gageidx]),:] = attribute_norm2[np.array(merit_idx[gage]).astype(int),:]
                
                
            #     attr2 = np.expand_dims(attr2, axis=1)
            #     attr2 = np.repeat(attr2, xTest2.shape[1], axis=1)
            #     xTest2 = np.concatenate([xTest2, attr2], 2) 
                
            #     xTest2_torch = torch.from_numpy(np.swapaxes(xTest2, 0,1)).to(xTest)
            #     yP = model(xTest, zTest,xTest2_torch, Ai_batch, Ac_batch, idx_matric)                              


            elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v2 ]:
            
                Merit_all,Ac_all,Ai_all =water_loss_info

            
                COMID_batch = []
                gage_area_batch = []
                gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
                for gage in gage_key_batch:
                    gage_area_batch.append(np.array(area_info[gage]['unitarea']).sum())
                    COMIDs = area_info[gage]['COMID']
                    COMID_batch.extend(COMIDs)
                COMID_batch_unique = list(set(COMID_batch))
                COMID_batch_unique.sort()       
                [_, Ind_batch, SubInd_batch] = np.intersect1d(COMID_batch_unique, Merit_all, return_indices=True)  
                forcing_norm2,attribute_norm2 = z_waterloss
                xTest2 = forcing_norm2[SubInd_batch,:,:]
                attr2 = attribute_norm2[SubInd_batch,:]

                attr2 = np.expand_dims(attr2, axis=1)
                attr2 = np.repeat(attr2, xTest2.shape[1], axis=1)
                xTest2 = np.concatenate([xTest2, attr2], 2) 
                
                
                Ai_batch = Ai_all[SubInd_batch]
                Ac_batch = Ac_all[SubInd_batch]
                print(len(Ac_batch)," merits in this batch")
                idx_matric = np.zeros((len(COMID_batch_unique),(iE[i]-iS[i])))                    
                for ii, gage in enumerate(gage_key_batch):
                    COMIDs = area_info[gage]['COMID']                        
                    [_, _,  SubInd] = np.intersect1d(COMIDs, np.array(COMID_batch_unique)[Ind_batch], return_indices=True)
                    idx_matric[SubInd,ii] = 1/gage_area_batch[ii]

                xTest2_torch = torch.from_numpy(np.swapaxes(xTest2, 0,1)).to(xTest)

                
                yP = model(xTest, zTest,xTest2_torch, Ai_batch, Ac_batch, idx_matric)

            elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v4 ]:
            
                Merit_all,Ac_all,Ai_all =water_loss_info

            
                COMID_batch = []
                gage_area_batch = []
                gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
                for gage in gage_key_batch:
                    gage_area_batch.append(np.array(area_info[gage]['unitarea']).sum())
                    COMIDs = area_info[gage]['COMID']
                    COMID_batch.extend(COMIDs)
                COMID_batch_unique = list(set(COMID_batch))
                COMID_batch_unique.sort()       
                [_, Ind_batch, SubInd_batch] = np.intersect1d(COMID_batch_unique, Merit_all, return_indices=True)  
                _,attribute_norm2 = z_waterloss
                attr2 = attribute_norm2[SubInd_batch,:]


                Ai_batch = Ai_all[SubInd_batch]
                Ac_batch = Ac_all[SubInd_batch]
                print(len(Ac_batch)," merits in this batch")
                idx_matric = np.zeros((len(COMID_batch_unique),(iE[i]-iS[i])))                    
                for ii, gage in enumerate(gage_key_batch):
                    COMIDs = area_info[gage]['COMID']                        
                    [_, _,  SubInd] = np.intersect1d(COMIDs, np.array(COMID_batch_unique)[Ind_batch], return_indices=True)
                    idx_matric[SubInd,ii] = 1/gage_area_batch[ii]

                xTest2_torch = torch.from_numpy(attr2).to(xTest)

                
                yP = model(xTest, zTest,xTest2_torch, Ai_batch, Ac_batch, idx_matric)


               
            else:
                yP = model(xTest, zTest)
        if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            yP = model(xTest, zTest)
        if type(model) in [cnn.LstmCnn1d]:
            xTest = xTest.permute(1, 2, 0)
            yP = model(xTest)
            yP = yP.permute(2, 0, 1)

        # CP-- marks the beginning of problematic merge
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        if doMC is not False:
            yOutMC = ySS.swapaxes(0, 1)

        # save output
        for k in range(ny):
            f = fLst[k]
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)
        if doMC is not False:
            for k in range(ny):
                f = fLst[ny+k]
                pd.DataFrame(yOutMC[:, :, k]).to_csv(
                    f, header=False, index=False)

        # model.zero_grad()
        torch.cuda.empty_cache()

    for f in fLst:
        f.close()



def testModel_large_river(model, x,z_waterloss, c, *,  area_info =None,  gage_key = None, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x) is tuple or type(x) is list:
        x, z = x
        if type(model) is rnn.CudnnLstmModel:
            # For Cudnn, only one input. First concat inputs and obs
            x = np.concatenate([x, z], axis=2)
            z = None
    else:
        z = None
    _, nt, nx = x.shape
    ngrid = len(gage_key)
    if c is not None:
        nc = c.shape[-1]
    if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v6]:
        ny=6 # streamflow
    elif type(model) in [rnn.MultiInv_SACSMAModel]:
        ny=2 # streamflow
    else:
        ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    torch.set_grad_enabled(False)

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

        if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v6]:

            gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
            Ai_batch = []
            Ac_batch = []
            id_list = []
            start_id = 0
            for gage in gage_key_batch:
                unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                uparea = area_info[gage]['uparea']
                Ai_batch.extend(unitarea)
                Ac_batch.extend(uparea)
                id_list.append(range(start_id, start_id+len(unitarea)))
                start_id = start_id+len(unitarea)


            idx_matric = np.zeros((len(Ai_batch),(iE[i]-iS[i])))
            for ii in range(iE[i]-iS[i]):
                idx_matric[np.array(id_list[ii]),ii] = 1

            forcing_norm2,attribute_norm2 = z_waterloss    

            xTest_torch = torch.from_numpy(np.swapaxes(x, 0,1)).float().cuda()
            attr_torch = torch.from_numpy(attribute_norm2).float().cuda()

            attr2_expand = np.repeat(np.expand_dims(attribute_norm2, axis=1), x.shape[1], axis=1)
            #zTest2 = np.concatenate((forcing_norm2,attr2_expand),axis = -1)
            #zTest2_torch = torch.from_numpy(np.swapaxes(zTest2, 0, 1)).float().cuda()
            routpara = model(xTest_torch, None,attr_torch, Ai_batch, Ac_batch, idx_matric) 




        pararout = routpara.detach().cpu().numpy()

    return pararout







        # save output
    #     for k in range(yOut.shape[-1]):
    #         f = fLst[k]
    #         pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)

    #     # model.zero_grad()
    #     torch.cuda.empty_cache()

    # for f in fLst:
    #     f.close()



def testModel_merit(model, x,z_waterloss, c, *, water_loss_info = None, area_info =None,  gage_key = None, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    x, z = x
    ny=6 # streamflow

    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    torch.set_grad_enabled(False)

    ngrid = len(gage_key)
    nmerit_list = []
    meritid = 0
   
    for gage in gage_key:
   
        meritid = meritid + len(area_info[gage]['COMID'])
        nmerit_list.append(meritid)
    iS = [0]
    previous_nmerit = 0
    merit_interval = batchSize

    for gageIndex, number_merit in enumerate(nmerit_list):
        if (number_merit-previous_nmerit) >=merit_interval:
            iS.append(gageIndex)
            previous_nmerit = nmerit_list[gageIndex-1]

    iS = np.array(iS)
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

        if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v5 ]:
        
            Merit_all,Ac_all,Ai_all =water_loss_info
        
            COMID_batch = []
            gage_area_batch = []
            gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
            for gage in gage_key_batch:
                gage_area_batch.append(np.array(area_info[gage]['unitarea']).sum())
                COMIDs = area_info[gage]['COMID']
                COMID_batch.extend(COMIDs)
            COMID_batch_unique = list(set(COMID_batch))
            COMID_batch_unique.sort()       
            [_, Ind_batch, SubInd_batch] = np.intersect1d(COMID_batch_unique, Merit_all, return_indices=True)  
            forcing_norm2,attribute_norm2 = z_waterloss
            attr2 = attribute_norm2[SubInd_batch,:]


            Ai_batch = Ai_all[SubInd_batch]
            Ac_batch = Ac_all[SubInd_batch]
            
            xTest = x[SubInd_batch,:,:]
            xTest2 = forcing_norm2[SubInd_batch,:,:]
            print(len(Ac_batch)," merits in this batch")
            idx_matric = np.zeros((len(COMID_batch_unique),(iE[i]-iS[i])))                    
            for ii, gage in enumerate(gage_key_batch):
                COMIDs = area_info[gage]['COMID']                        
                [_, _,  SubInd] = np.intersect1d(COMIDs, np.array(COMID_batch_unique)[Ind_batch], return_indices=True)
                idx_matric[SubInd,ii] = 1/gage_area_batch[ii]

            att2_torch = torch.from_numpy(attr2).float().cuda()
            xTest_torch = torch.from_numpy(np.swapaxes(xTest, 0,1)).float().cuda()
            xTest2_torch = torch.from_numpy(np.swapaxes(xTest2, 0,1)).float().cuda()
            
            yP = model(xTest_torch, xTest2_torch,att2_torch, Ai_batch, Ac_batch, idx_matric)

        if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v6 ]:
        
            Merit_all,Ac_all,Ai_all =water_loss_info
        
            COMID_batch = []
            gage_area_batch = []
            gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
            for gage in gage_key_batch:
                gage_area_batch.append(np.array(area_info[gage]['unitarea']).sum())
                COMIDs = area_info[gage]['COMID']
                COMID_batch.extend(COMIDs)
            COMID_batch_unique = list(set(COMID_batch))
            COMID_batch_unique.sort()       
            [_, Ind_batch, SubInd_batch] = np.intersect1d(COMID_batch_unique, Merit_all, return_indices=True)  
            forcing_norm2,attribute_norm2 = z_waterloss
            attr2 = attribute_norm2[SubInd_batch,:]


            Ai_batch = Ai_all[SubInd_batch]
            Ac_batch = Ac_all[SubInd_batch]
            
            xTest = x[SubInd_batch,:,:]
            xTest2 = forcing_norm2[SubInd_batch,:,:]
            print(len(Ac_batch)," merits in this batch")
            idx_matric = np.zeros((len(COMID_batch_unique),(iE[i]-iS[i])))                    
            for ii, gage in enumerate(gage_key_batch):
                COMIDs = area_info[gage]['COMID']                        
                [_, _,  SubInd] = np.intersect1d(COMIDs, np.array(COMID_batch_unique)[Ind_batch], return_indices=True)
                idx_matric[SubInd,ii] = 1/gage_area_batch[ii]


            xTest_torch = torch.from_numpy(np.swapaxes(xTest, 0,1)).float().cuda()
            attr_torch = torch.from_numpy(attr2).float().cuda()

            attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTest.shape[1], axis=1)
            zTest2 = np.concatenate((xTest2,attr2_expand),axis = -1)
            zTest2_torch = torch.from_numpy(np.swapaxes(zTest2, 0, 1)).float().cuda()
            yP = model(xTest_torch, zTest2_torch,attr_torch, Ai_batch, Ac_batch, idx_matric) 


        if type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v8 ]:
        
            Merit_all,Ac_all,Ai_all =water_loss_info
          
            COMID_batch = []
            gage_area_batch = []
            gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
            for gage in gage_key_batch:
                gage_area_batch.append(np.array(area_info[gage]['unitarea']).sum())
                COMIDs = area_info[gage]['COMID']
                COMID_batch.extend(COMIDs)
            COMID_batch_unique = list(set(COMID_batch))
            COMID_batch_unique.sort()       
            [_, Ind_batch, SubInd_batch] = np.intersect1d(COMID_batch_unique, Merit_all, return_indices=True)  
            forcing_norm2,attribute_norm2,norm_gage_area = z_waterloss
            attr2 = attribute_norm2[SubInd_batch,:]

            norm_gage_area_batch = norm_gage_area[iS[i]:iE[i],0]
            Ai_batch = Ai_all[SubInd_batch]
            Ac_batch = Ac_all[SubInd_batch]
            
            xTest = x[SubInd_batch,:,:]
            xTest2 = forcing_norm2[SubInd_batch,:,:]
            print(len(Ac_batch)," merits in this batch")
            idx_matric = np.zeros((len(COMID_batch_unique),(iE[i]-iS[i]))) 
            
            attr_rout = np.full((len(gage_key_batch),attr2.shape[-1]),np.nan)
                               
            for ii, gage in enumerate(gage_key_batch):
                COMIDs = area_info[gage]['COMID']                        
                [_, _,  SubInd] = np.intersect1d(COMIDs, np.array(COMID_batch_unique)[Ind_batch], return_indices=True)
                idx_matric[SubInd,ii] = 1/gage_area_batch[ii]
                weights =  Ai_batch[SubInd]/gage_area_batch[ii]
                attr_rout[ii,:] =np.sum(attr2[SubInd,:] * weights[:,np.newaxis], axis = 0) 
                attr_rout[ii,-1] = norm_gage_area_batch[ii]                

            xTest_torch = torch.from_numpy(np.swapaxes(xTest, 0,1)).float().cuda()
            attr_torch = torch.from_numpy(attr2).float().cuda()
            attr_rout_torch = torch.from_numpy(attr_rout).float().cuda()
            attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTest.shape[1], axis=1)
            zTest2 = np.concatenate((xTest2,attr2_expand),axis = -1)
            zTest2_torch = torch.from_numpy(np.swapaxes(zTest2, 0, 1)).float().cuda()
            yP = model(xTest_torch, zTest2_torch,attr_torch, attr_rout_torch,Ai_batch, Ac_batch, idx_matric) 

        # CP-- marks the beginning of problematic merge
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)


        # save output
        for k in range(yOut.shape[-1]):
            f = fLst[k]
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)

        # model.zero_grad()
        torch.cuda.empty_cache()

    for f in fLst:
        f.close()



def testModel_multiGPU(model, x ,zTest2, area_info,gage_key, region_number,time_range,variables_name,folderpath, c, device,*, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x) is tuple or type(x) is list:
        x, z = x
        if type(model) is rnn.CudnnLstmModel:
            # For Cudnn, only one input. First concat inputs and obs
            x = np.concatenate([x, z], axis=2)
            z = None
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    if type(model) in [rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel,rnn.MultiInv_HBVTDModel_water_loss,rnn.MultiInv_HBVTDModel_water_loss_v2,rnn.MultiInv_HBVTDModel_water_loss_v4,rnn.MultiInv_HBVTDModel_water_loss_v5,rnn.MultiInv_HBVTDModel_water_loss_v6]:
        ny=6 # streamflow
    elif type(model) in [rnn.MultiInv_SACSMAModel]:
        ny=2 # streamflo
    else:
        ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.to(device)

    model.train(mode=False)
    torch.set_grad_enabled(False)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct
    # yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)

    # deal with file name to save
    if filePathLst is None:
        filePathLst = ['out' + str(x) for x in range(ny)]
    # fLst = list()
    # for filePath in filePathLst:
    #     if os.path.exists(filePath):
    #         os.remove(filePath)
    #     f = open(filePath, 'a')
    #     fLst.append(f)

    # forward for each batch
    for i in range(0, len(iS)):
        print('batch ', i, "on device ", device, "for zone ", region_number)
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
            if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                if len(z.shape) == 2:
                    # Used for local calibration kernel as FDC
                    # x = Ngrid * Ntime
                    zTest = torch.from_numpy(z[iS[i]:iE[i], :]).float()
                elif len(z.shape) == 3:
                    # used for LC-SMAP x=Ngrid*Ntime*Nvar
                    zTest = torch.from_numpy(np.swapaxes(z[iS[i]:iE[i], :, :], 1, 2)).float()
            else:
                zTemp = z[iS[i]:iE[i], :, :]
                # if type(model) in [rnn.CudnnInvLstmModel]: # Test SMAP Inv with attributes
                #     cInv = np.repeat(
                #         np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), zTemp.shape[1], axis=1)
                #     zTemp = np.concatenate([zTemp, cInv], 2)
                zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
            if torch.cuda.is_available():
                zTest = zTest.to(device)
        if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
            # if z is not None:
            #     xTest = torch.cat((xTest, zTest), dim=2)
            yP = model(xTest)
            if doMC is not False:
                ySS = np.zeros(yP.shape)
                yPnp=yP.detach().cpu().numpy()
                for k in range(doMC):
                    # print(k)
                    yMC = model(xTest, doDropMC=True).detach().cpu().numpy()
                    ySS = ySS+np.square(yMC-yPnp)
                ySS = np.sqrt(ySS)/doMC
        if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                           rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel, rnn.CudnnInvLstmModel,
                           rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel,rnn.MultiInv_SACSMAModel,rnn.MultiInv_HBVTDModel_water_loss,rnn.MultiInv_HBVTDModel_water_loss_v2,rnn.MultiInv_HBVTDModel_water_loss_v4,rnn.MultiInv_HBVTDModel_water_loss_v5,rnn.MultiInv_HBVTDModel_water_loss_v6]:
            if type(model) in [rnn.MultiInv_HBVTDModel_water_loss]:
            
                gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
              
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
    
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)


                idx_matric = np.zeros((len(Ai_batch),(iE[i]-iS[i])))
                for ii in range(iE[i]-iS[i]):
                    idx_matric[np.array(id_list[ii]),ii] = 1

                yP = model(xTest, zTest, Ai_batch, Ac_batch, idx_matric)



            elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v2 ]:    
                gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)


                idx_matric = np.zeros((len(Ai_batch),(iE[i]-iS[i])))
                for ii in range(iE[i]-iS[i]):
                    idx_matric[np.array(id_list[ii]),ii] = 1

                zTest2_torch = torch.from_numpy(np.swapaxes(zTest2[iS[i]:iE[i]], 0,1)).to(zTest)
                yP = model(xTest, zTest,zTest2_torch, Ai_batch, Ac_batch, idx_matric)  

            elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v4 ]:    
                gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)


                idx_matric = np.zeros((len(Ai_batch),(iE[i]-iS[i])))
                for ii in range(iE[i]-iS[i]):
                    idx_matric[np.array(id_list[ii]),ii] = 1

                zTest2_torch = torch.from_numpy(zTest2[iS[i]:iE[i]]).to(zTest)
                yP = model(xTest, zTest,zTest2_torch, Ai_batch, Ac_batch, idx_matric)  

            elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v5 ]:    
                gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)


                idx_matric = np.zeros((len(Ai_batch),(iE[i]-iS[i])))
                for ii in range(iE[i]-iS[i]):
                    idx_matric[np.array(id_list[ii]),ii] = 1

                zTest2_torch = torch.from_numpy(zTest2[iS[i]:iE[i]]).to(zTest)
                yP = model(xTest, zTest,zTest2_torch, Ai_batch, Ac_batch, idx_matric)  




            elif type(model) in [rnn.MultiInv_HBVTDModel_water_loss_v6 ]:    
                gage_key_batch = np.array(gage_key)[iS[i]:iE[i]]
                Ai_batch = []
                Ac_batch = []
                id_list = []
                start_id = 0
                for gage in gage_key_batch:
                    unitarea = area_info[gage]['unitarea']/np.array(area_info[gage]['unitarea']).sum()
                    uparea = area_info[gage]['uparea']
                    Ai_batch.extend(unitarea)
                    Ac_batch.extend(uparea)
                    id_list.append(range(start_id, start_id+len(unitarea)))
                    start_id = start_id+len(unitarea)


                idx_matric = np.zeros((len(Ai_batch),(iE[i]-iS[i])))
                for ii in range(iE[i]-iS[i]):
                    idx_matric[np.array(id_list[ii]),ii] = 1



                zTest2_torch = torch.from_numpy(zTest2[iS[i]:iE[i]]).to(zTest)
                yP = model(xTest, zTest,zTest2_torch, Ai_batch, Ac_batch, idx_matric)  







            else:
                yP = model(xTest, zTest)
        if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            yP = model(xTest, zTest)
        if type(model) in [cnn.LstmCnn1d]:
            xTest = xTest.permute(1, 2, 0)
            yP = model(xTest)
            yP = yP.permute(2, 0, 1)
        model.zero_grad()
        torch.cuda.empty_cache()
        # CP-- marks the beginning of problematic merge
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        if doMC is not False:
            yOutMC = ySS.swapaxes(0, 1)

        # save output
        if i==0:
            yOutAll = yOut
            if doMC is not False:
                yOutMCAll = yOutMC
        else:
            yOutAll = np.concatenate((yOutAll,yOut),axis = 0)
            if doMC is not False:
                yOutMCAll = np.concatenate((yOutMCAll, yOutMC), axis=-1)
    data_arrays = {}
    #data_x = np.full((COMID.shape[0],self.time_range.shape[0],len(self.forcing)),np.nan)

    for idx, var_x in enumerate(variables_name):
        
        data_array = xr.DataArray(
            yOutAll[:,:,idx],
            dims = ['COMID','time'],
            coords = {'COMID':gage_key,
                     'time':time_range}
        )

        data_arrays[var_x] = data_array

    xr_dataset = xr.Dataset(data_arrays)
    xr_dataset.to_zarr(store=folderpath, group=f'{region_number}', mode='w')






def visualParameters(model, x, c, *, batchSize=None, filePathLst=None, BufferLenth = 0):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x) is tuple or type(x) is list:
        x, z = x
        if type(model) is rnn.CudnnLstmModel:
            # For Cudnn, only one input. First concat inputs and obs
            x = np.concatenate([x, z], axis=2)
            z = None
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    if type(model) in [rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel]:
        ny=5 # streamflow
    else:
        ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    torch.set_grad_enabled(False)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct
    # yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)


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
            xTest = xTest.cuda()
        if z is not None:
            if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                if len(z.shape) == 2:
                    # Used for local calibration kernel as FDC
                    # x = Ngrid * Ntime
                    zTest = torch.from_numpy(z[iS[i]:iE[i], :]).float()
                elif len(z.shape) == 3:
                    # used for LC-SMAP x=Ngrid*Ntime*Nvar
                    zTest = torch.from_numpy(np.swapaxes(z[iS[i]:iE[i], :, :], 1, 2)).float()
            else:
                zTemp = z[iS[i]:iE[i], :, :]
                # if type(model) in [rnn.CudnnInvLstmModel]: # Test SMAP Inv with attributes
                #     cInv = np.repeat(
                #         np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), zTemp.shape[1], axis=1)
                #     zTemp = np.concatenate([zTemp, cInv], 2)
                zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
            if torch.cuda.is_available():
                zTest = zTest.cuda()
        yP,hbvpara,routpara = model(xTest, zTest)
        nt = hbvpara.shape[0]
        parhbvFull = hbvpara[-1, :, :, :].unsqueeze(0).repeat([nt, 1, 1, 1])  # static para matrix

        # create probability mask for each parameter on the basin dimension
        tdlst=[1,13]
        for ix in tdlst:

            parhbvFull[:, :, ix-1, :] = hbvpara[:, :, ix-1, :]

        #
        #
        # hbvscaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
        #                 [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0.3,5]]  # HBV para
        # routscaLst = [[0, 2.9], [0, 6.5]]
        # for ip in range(len(hbvscaLst)): # not include routing. Scaling the parameters
        #     hbvpara[:,:,ip,:] = hbvscaLst[ip][0] + hbvpara[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        # hbvpara = hbvpara.mean(dim = -1)
        parahbv = parhbvFull.detach().cpu().numpy()
        parahbv = parahbv[BufferLenth:,:,:,:].mean(0).mean(-1)
        pararout = routpara.detach().cpu().numpy()

        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        yOut = yOut[:,BufferLenth:,:]
        if i == 0:
            allBasinParaHBV = parahbv
            allBasinParaRout = pararout
            yOutAll = yOut
        else:
            allBasinParaHBV = np.concatenate((allBasinParaHBV,parahbv),axis = 0)
            allBasinParaRout = np.concatenate((allBasinParaRout, pararout), axis=0)
            yOutAll = np.concatenate((yOutAll, yOut), axis=0)


    #path_model = '/data/yxs275/NROdeSolver/output/HBVtest_module_hbv_1_13_dynamic_rout_static/'
    # paraHBVFile  = path_model+"discrete_static_paraHBV.npy"
    # paraRoutFile = path_model + "discrete_static_paraRout.npy"
    np.save(filePathLst[0],allBasinParaHBV)
    np.save(filePathLst[1], allBasinParaRout)
    np.save(filePathLst[2], yOutAll)
    return yOutAll,allBasinParaHBV,allBasinParaRout





def selectSubset_multiGPUs(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0,device = None):
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
                xTensor = xTensor.to(device)
                cTensor = cTensor.to(device)
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.to(device)
    return out

def reduce_sum_until_target(array,indices, target_sum=3000,batchsize = 100):
    # Randomly select 100 integers from the array
    
    selected = array[indices]
    
    # If the sum of the selected integers is greater than 3000
    if np.sum(selected) > target_sum:
        # Find the index of the largest value
        max_idx = np.argmax(selected)
        max_value_idx = indices[max_idx]
        
        # Create a list of indices excluding the largest one
        other_indices = np.delete(indices, max_idx)
        
        # Shuffle the other indices randomly
        np.random.shuffle(other_indices)
        
        # Start removing elements randomly until the sum is less than or equal to 3000
        current_sum = np.sum(selected)
        i = 0
        while current_sum > target_sum and i < len(other_indices):
            current_sum -= array[other_indices[i]]
            other_indices[i] = -1  # Mark as removed by setting to -1
            i += 1
        
        # Combine the largest value index with unremoved indices
        remaining_indices = np.concatenate(([max_value_idx], other_indices[other_indices >= 0]))
    else:
        remaining_indices = indices

    return remaining_indices


def testModelCnnCond(model, x, y, *, batchSize=None):
    ngrid, nt, nx = x.shape
    ct = model.ct
    ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
    # cTest = torch.from_numpy(np.swapaxes(y[:, 0:ct, :], 1, 0)).float()
    cTest = torch.zeros([ct, ngrid, y.shape[-1]], requires_grad=False)
    for k in range(ngrid):
        ctemp = y[k, 0:ct, 0]
        i0 = np.where(np.isnan(ctemp))[0]
        i1 = np.where(~np.isnan(ctemp))[0]
        if len(i1) > 0:
            ctemp[i0] = np.interp(i0, i1, ctemp[i1])
            cTest[:, k, 0] = torch.from_numpy(ctemp)

    if torch.cuda.is_available():
        xTest = xTest.cuda()
        cTest = cTest.cuda()
        model = model.cuda()

    model.train(mode=False)

    yP = torch.zeros([nt - ct, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)
    for i in range(0, len(iS)):
        xTemp = xTest[:, iS[i]:iE[i], :]
        cTemp = cTest[:, iS[i]:iE[i], :]
        yP[:, iS[i]:iE[i], :] = model(xTemp, cTemp)
    yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
    return yOut


def randomSubset(x, y, dimSubset):
    ngrid, nt, nx = x.shape
    batchSize, rho = dimSubset
    xTensor = torch.zeros([rho, batchSize, x.shape[-1]], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, y.shape[-1]], requires_grad=False)
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    for k in range(batchSize):
        temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        yTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT


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

