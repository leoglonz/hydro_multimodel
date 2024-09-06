from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

import scipy.stats

def statError(pred, target):
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)

    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
    # ubRMSE
    #dMax_rel = (np.nanmax(pred,axis=1)-np.nanmax(target,axis=1))/np.nanmax(target,axis=1)
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
    # FDC metric
    # predFDC = calFDC(pred)
    # targetFDC = calFDC(target)
    #FDCRMSE = np.sqrt(np.nanmean((predFDC - targetFDC) ** 2, axis=1))
    # rho R2 NSE
    dMax_rel = np.full(ngrid, np.nan)
    dMax = np.full(ngrid, np.nan)
    Corr = np.full(ngrid, np.nan)
    Bias_rel = np.full(ngrid, np.nan)
    CorrSp = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    NNSE = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    PBiasother = np.full(ngrid, np.nan)
    absPBiaslow = np.full(ngrid, np.nan)
    absPBiashigh = np.full(ngrid, np.nan)
    absPBias = np.full(ngrid, np.nan)
    absPBiasother = np.full(ngrid, np.nan)
    KGE = np.full(ngrid, np.nan)
    KGE12 = np.full(ngrid, np.nan)
    RMSElow = np.full(ngrid, np.nan)
    RMSEhigh = np.full(ngrid, np.nan)
    RMSEother = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]


        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]


        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            maxobs = np.nanmax(yy)
            maxIdx = np.nanargmax(yy)
            window_lower = 10
            window_upper = 11
            if (maxIdx < window_lower):
                window_lower = maxIdx
            elif (window_upper > len(xx) - maxIdx):
                window_upper = len(xx) - maxIdx

            maxpred = np.nanmax(xx[maxIdx - window_lower:maxIdx + window_upper])
            #  maxpred = np.nanmax(x)
            dMax[k] = maxpred - maxobs
            dMax_rel[k] = (maxpred - maxobs) / maxobs * 100


            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100

            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.98 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            otherpred = pred_sort[indexlow:indexhigh]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            othertarget = target_sort[indexlow:indexhigh]
            PBiaslow[k] = np.sum((lowpred - lowtarget)) / (np.sum(lowtarget) +0.0001)* 100
            PBiashigh[k] = np.sum((highpred - hightarget) )/ np.sum(hightarget) * 100
            PBiasother[k] = np.sum((otherpred - othertarget)) / np.sum(othertarget) * 100
            absPBiaslow[k] = np.sum(abs(lowpred - lowtarget)) / ((np.sum(lowtarget) +0.0001))* 100
            absPBiashigh[k] = np.sum(abs(highpred - hightarget) )/ np.sum(hightarget) * 100
            absPBiasother[k] = np.sum(abs(otherpred - othertarget)) / np.sum(othertarget) * 100
            Bias_rel[k] = (np.sum(xx)-np.sum(yy))/np.sum(yy)

            RMSElow[k] = np.sqrt(np.nanmean((lowpred - lowtarget)**2))
            RMSEhigh[k] = np.sqrt(np.nanmean((highpred - hightarget)**2))
            RMSEother[k] = np.sqrt(np.nanmean((otherpred - othertarget)**2))

            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                CorrSp[k] = scipy.stats.spearmanr(xx, yy)[0]
                yymean = yy.mean()
                yystd = np.std(yy)
                xxmean = xx.mean()
                xxstd = np.std(xx)
                KGE[k] = 1 - np.sqrt((Corr[k]-1)**2 + (xxstd/yystd-1)**2 + (xxmean/yymean-1)**2)
                KGE12[k] = 1 - np.sqrt((Corr[k] - 1) ** 2 + ((xxstd*yymean)/ (yystd*xxmean) - 1) ** 2 + (xxmean / yymean - 1) ** 2)
                SST = np.sum((yy-yymean)**2)
                SSReg = np.sum((xx-yymean)**2)
                SSRes = np.sum((yy-xx)**2)
                R2[k] = 1-SSRes/SST
                NSE[k] = 1-SSRes/SST
                NNSE[k] = SST/(SSRes+SST)

    outDict = dict(Bias=Bias,Bias_rel=Bias_rel, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, CorrSp=CorrSp, R2=R2, NSE=NSE,NNSE=NNSE,
                   FLV=PBiaslow, FHV=PBiashigh, PBias=PBias, PBiasother=PBiasother, absFLV=absPBiaslow, absFHV=absPBiashigh, absPBias=absPBias, absPBiasother=absPBiasother, KGE=KGE, KGE12=KGE12,
                   lowRMSE=RMSElow, highRMSE=RMSEhigh, midRMSE=RMSEother,rdMax = dMax_rel,dMax = dMax)
    return outDict


#path_to_zarr = "/data/tkb5476/projects/dMC-dev/runs/dMCv0.1-srb_validation-dpl_v3/2024-02-08_14-09-48/1985-10-04_1995-09-30"
#path_to_zarr = "/data/tkb5476/projects/dMC-dev/runs/large_basins/zone_77/1981-10-04_1995-09-30/"
path_to_zarr = "/data/yxs275/FromOthers/1981-10-04_1995-09-30"
ds = xr.open_zarr(path_to_zarr)
gage_ids = ds.gage_ids.data
daily_pred = ds.predictions.resample(time='1D').mean().values
daily_obs = ds.observations.resample(time='1D').mean().values
time = pd.date_range('1981-10-04',f'1995-09-30', freq='d')

data_folder = "/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/gages/dataCONUS3200/"

results_folder = "/data/yxs275/DPL_HBV/CONUS_3200_Output/dPL_local_daymet_filled_NaN/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/"
with open(data_folder+'train_data_dict.json') as f:
    train_data_dict = json.load(f)
gageID = train_data_dict['sites_id']

forward_pref = np.load(results_folder+ "validation_ypred_dHBV.npy")


forward_time = pd.date_range(train_data_dict['t_final_range'][0],train_data_dict['t_final_range'][1], freq='d')

obs_all = np.load(data_folder+"train_flow.npy")

attributeGAGEII  = np.load(data_folder+"train_attr.npy")
attributeGAGEIILst  = train_data_dict['constant_cols']
basinAreaName = "DRAIN_SQKM"
basin_area = attributeGAGEII[:,np.where(np.array(attributeGAGEIILst)=="DRAIN_SQKM")[0]]
newgage_ids = []
gage_idx = []
skipped_idx = []
for i in range(len(gage_ids)):
    if len(gage_ids[i])<9:
        new_id = gage_ids[i]
        new_id = str(new_id).zfill(8)
    newgage_ids.append(new_id)
    try:
        gage_idx.append(np.where(np.array(gageID)==new_id)[0][0])
    except:
        print(new_id, "does not have simulations")
        skipped_idx.append(i)

basin_area_selected = basin_area[gage_idx]
forward_pref_selected = forward_pref[gage_idx,forward_time.get_loc(time[0]):forward_time.get_loc(time[-1])+1]* (basin_area_selected *1000)/(24*60*60)

obs_selected = obs_all[gage_idx,forward_time.get_loc(time[0]):forward_time.get_loc(time[-1])+1,0]* 0.0283168

for idx in range(len(gage_idx)):
    if idx in skipped_idx:
        continue
    else:
        predAllar = np.concatenate((np.expand_dims(forward_pref_selected[idx,:], axis = 0),np.expand_dims(daily_pred[idx,:], axis = 0)), axis = 0)
        obsAll = np.concatenate((np.expand_dims(obs_selected[idx,:], axis = 0),np.expand_dims(obs_selected[idx,:], axis = 0)), axis = 0)
        obsAll_tadd = np.concatenate(
            (np.expand_dims(daily_obs[idx, :], axis=0), np.expand_dims(daily_obs[idx, :], axis=0)), axis=0)

        evaDict = [statError(predAllar[:,:], obsAll_tadd[:,:])]
        evaDictLst = evaDict
        keyLst = ['NSE', 'Bias','Corr','Bias_rel','NNSE']
        dataBox = list()
        for iS in range(len(keyLst)):
            statStr = keyLst[iS]
            temp = list()
            for k in range(len(evaDictLst)):
                data = evaDictLst[k][statStr]
                #data = data[~np.isnan(data)]
                temp.append(data)
            dataBox.append(temp)

        print("NSE from Tadd's observation :", dataBox[0][0])

        evaDict = [statError(predAllar[:,:], obsAll[:,:])]
        evaDictLst = evaDict
        keyLst = ['NSE', 'Bias','Corr','Bias_rel','NNSE']
        dataBox = list()
        for iS in range(len(keyLst)):
            statStr = keyLst[iS]
            temp = list()
            for k in range(len(evaDictLst)):
                data = evaDictLst[k][statStr]
                #data = data[~np.isnan(data)]
                temp.append(data)
            dataBox.append(temp)




        NSElst_ar =dataBox[0][0]
        correlationslst_ar  =dataBox[2][0]



        # idx = idx_ar[23]
        fontsize = 18
        plt.rcParams.update({'font.size': fontsize})
        plotTime = pd.date_range('1985-10-05', f'1995-09-25', freq='d')

        plt.figure(figsize=(10, 6))

        plt.plot(time,
                 forward_pref_selected[idx,:]   ,
                 label=f'Forward flow: NSE {round(NSElst_ar[0],2)}', lw=2, color='red')
        plt.plot(time,
                 daily_pred[idx,:],
                 label=f'Routed flow: NSE {round(NSElst_ar[1],2)}', lw=2, color='blue')

        plt.plot(time,
                 obs_selected[idx,:], label='Observation', lw=2,
                 color='k')
        plt.title(f'Gage {newgage_ids[idx]}, Catchment area {basin_area_selected[idx]} km$^2$')
        plt.xlabel('Date')
        plt.ylabel(r'Discharge (m$^3$/s)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=True)