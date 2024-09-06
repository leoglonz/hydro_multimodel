import sys
from pathlib import Path

# Construct an absolute path by going up two directories from this script's location
absolute_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(absolute_path))
from hydroDL.data import scale
from hydroDL.post import plot, stat
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle

attribute_file = '/projects/mhpi/yxs275/Data/attributes_haoyu/attributes_haoyu.csv'
shapeID_str_lst= np.load("/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/shapeID_str_lst.npy")

test_span = pd.date_range('1995-10-01',f'2010-09-30', freq='d')

attributeALL_df = pd.read_csv(attribute_file,index_col=0)
attributeALL_df = attributeALL_df.sort_values(by='id')

attributeAllLst = attributeALL_df.columns

basin_area = np.expand_dims(attributeALL_df["area"].values,axis = 1)
idLst_new = attributeALL_df["id"].values

idLst_old = [int(id) for id in shapeID_str_lst]
[C, ind1, SubInd_id] = np.intersect1d(idLst_new, idLst_old, return_indices=True)
if(not (idLst_new==np.array(idLst_old)[SubInd_id]).all()):
   raise Exception("Ids of subset gage do not match with id in the attribtue file")


data_folder = "/projects/mhpi/yxs275/Data/generate_for_CONUS_3200/gages/dataCONUS3200/"
streamflow_test = np.load(data_folder+"test_flow.npy")
streamflow_test = streamflow_test[SubInd_id,:,:]

lat =  attributeALL_df["lat"].values[ind1]
lon=  attributeALL_df["lon"].values[ind1]
meanP=  attributeALL_df["meanP"].values[ind1]

attributeLst = ['area','ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']
forcing_HBV_List = ['prcp',  'tmax', 'tmin', ]
attributeLst_old = attributeLst.copy()


[C, _, SubInd] = np.intersect1d(attributeLst, attributeAllLst, return_indices=True)
attribute = attributeALL_df.iloc[ind1, np.sort(SubInd)].values

attributeLst  = list(attributeAllLst[np.sort(SubInd)])

if(not (np.array(attributeLst)==np.array(attributeLst_old)).all()):
   raise Exception("AttributeLst is not in the order provided")


streamflow_trans = scale._basin_norm(
                        streamflow_test[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )
test_obs = streamflow_trans[:,:,0]



A_change_list = [5,10,20,30,50,80,100,200,300,500,800,1000,1200,1500,1800,2000,3000,4000,5000,6000,8000,10000,15000,20000,30000,40000]


prediction_out = f"/projects/mhpi/yxs275/model/scale_analysis"


model_path = ['/dPL_local_daymet_new_attr_NSE_loss_wo_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365',
              '/dPL_local_daymet_new_attr_NSE_loss_w_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365',
              '/dPL_local_daymet_new_attr_RMSE_loss_wo_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365',
              '/dPL_local_daymet_new_attr_RMSE_loss_w_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365',]

mode = "/P_increase_0.01"
save_path = prediction_out+mode
if os.path.exists(save_path) is False:
    os.mkdir(save_path)

for pathidx, path in enumerate(model_path) :


    filePathLst = [prediction_out +path+ mode + f"/allBasinParaHBV_area_new_base", prediction_out +path+mode + f"/allBasinParaRout_area_new_base",prediction_out+path+mode +f"/allBasinFluxes_area_new_base"]
    if pathidx == 0:
        allBasinParaHBV_base = np.load(filePathLst[0]+".npy")
        allBasinParaRout_base = np.load(filePathLst[1]+".npy")
        allBasinFluxes_base = np.load(filePathLst[2]+".npy")[:,-streamflow_test.shape[1]:,:]
    else:
        allBasinParaHBV_base = allBasinParaHBV_base + np.load(filePathLst[0]+".npy")
        allBasinParaRout_base = allBasinParaRout_base + np.load(filePathLst[1]+".npy")
        allBasinFluxes_base = allBasinFluxes_base + np.load(filePathLst[2]+".npy")[:,-streamflow_test.shape[1]:,:]        


allBasinParaHBV_base = allBasinParaHBV_base/len(model_path)
allBasinParaRout_base = allBasinParaRout_base/len(model_path)
allBasinFluxes_base = allBasinFluxes_base/len(model_path)



filePathLst = [save_path + f"/allBasinParaHBV_area_new_base", save_path + f"/allBasinParaRout_area_new_base",save_path+f"/allBasinFluxes_area_new_base"]

np.save(filePathLst[0]+".npy",allBasinParaHBV_base)
np.save(filePathLst[1]+".npy",allBasinParaRout_base)
np.save(filePathLst[2]+".npy",allBasinFluxes_base)


dataPred = allBasinFluxes_base[:,:,0]

evaDict = [stat.statError(dataPred[:,:], test_obs)]
evaDictLst = evaDict
keyLst = ['NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))



for id_change in range(len(A_change_list)):

    for pathidx, path in enumerate(model_path) :


        filePathLst = [prediction_out +path+ mode + f"/allBasinParaHBV_area_new_{A_change_list[id_change]}", prediction_out +path+mode + f"/allBasinParaRout_area_new_{A_change_list[id_change]}",prediction_out+path+mode +f"/allBasinFluxes_area_new_{A_change_list[id_change]}"]
        if pathidx == 0:
            allBasinParaHBV_base = np.load(filePathLst[0]+".npy")
            allBasinParaRout_base = np.load(filePathLst[1]+".npy")
            allBasinFluxes_base = np.load(filePathLst[2]+".npy")[:,-streamflow_test.shape[1]:,:]
        else:
            allBasinParaHBV_base = allBasinParaHBV_base + np.load(filePathLst[0]+".npy")
            allBasinParaRout_base = allBasinParaRout_base + np.load(filePathLst[1]+".npy")
            allBasinFluxes_base = allBasinFluxes_base + np.load(filePathLst[2]+".npy")[:,-streamflow_test.shape[1]:,:]        


    allBasinParaHBV_base = allBasinParaHBV_base/len(model_path)
    allBasinParaRout_base = allBasinParaRout_base/len(model_path)
    allBasinFluxes_base = allBasinFluxes_base/len(model_path)



    filePathLst = [save_path + f"/allBasinParaHBV_area_new_{A_change_list[id_change]}", save_path + f"/allBasinParaRout_area_new_{A_change_list[id_change]}",save_path+f"/allBasinFluxes_area_new_{A_change_list[id_change]}"]

    np.save(filePathLst[0]+".npy",allBasinParaHBV_base)
    np.save(filePathLst[1]+".npy",allBasinParaRout_base)
    np.save(filePathLst[2]+".npy",allBasinFluxes_base)
