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
out = f"/projects/mhpi/yxs275/model/scale_analysis"
prediction_out = out+f"/scaling_analysis"
prediction_out_precp = out+f"/P_increase_0.01"
prediction_out_T = out+f"/T_increase_1"

if os.path.exists(prediction_out) is False:
    os.mkdir(prediction_out)


mean_delta_flow_frac_area_precp = np.full((len(attribute),len(A_change_list)),np.nan)
peak_delta_flow_area_precp = np.full((len(attribute),len(A_change_list)),np.nan)

mean_delta_flow_frac_area_T = np.full((len(attribute),len(A_change_list)),np.nan)
peak_delta_flow_area_T = np.full((len(attribute),len(A_change_list)),np.nan)

delta_Q7_area_precp = np.full((len(attribute),len(A_change_list)),np.nan)
delta_Q7_area_T = np.full((len(attribute),len(A_change_list)),np.nan)

def bias_meanflowratio_calc(pred,target):
    ngrid,nt = pred.shape
    Bias = np.full(ngrid, np.nan)
    meanflowratio = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            Bias[k] = (np.sum(xx)-np.sum(yy))/(np.sum(yy)+0.00001)
            meanflowratio[k]  = np.sum(xx)/(np.sum(yy)+0.00001)

    return Bias, meanflowratio

def calc_Q7(flow_data):
    flow_df = pd.DataFrame(flow_data.transpose())

    rolling_means = flow_df.rolling(window=7).mean()

    min_7_day_flows = rolling_means.min()
    return min_7_day_flows.values

for id_change in range(len(A_change_list)):
    print("Working on Area ", A_change_list[id_change])

    filePathLst = [prediction_out + f"/allBasinParaHBV_area_new_{A_change_list[id_change]}", prediction_out + f"/allBasinParaRout_area_new_{A_change_list[id_change]}",prediction_out+f"/allBasinFluxes_area_new_{A_change_list[id_change]}"]
    filePathLst_precp = [prediction_out_precp + f"/allBasinParaHBV_area_new_{A_change_list[id_change]}", prediction_out_precp + f"/allBasinParaRout_area_new_{A_change_list[id_change]}",prediction_out_precp+f"/allBasinFluxes_area_new_{A_change_list[id_change]}"]
    filePathLst_T = [prediction_out_T + f"/allBasinParaHBV_area_new_{A_change_list[id_change]}", prediction_out_T + f"/allBasinParaRout_area_new_{A_change_list[id_change]}",prediction_out_T+f"/allBasinFluxes_area_new_{A_change_list[id_change]}"]

    allBasinFluxes = np.load(filePathLst[2] + ".npy")[:,-len(test_span):,:]
    allBasinFluxes_precp = np.load(filePathLst_precp[2] + ".npy")[:, -len(test_span):, :]
    allBasinFluxes_T = np.load(filePathLst_T[2] + ".npy")[:, -len(test_span):, :]
    delta_precp = (allBasinFluxes_precp[:, :, 0] - allBasinFluxes[:, :, 0])
    delta_T = (allBasinFluxes_T[:, :, 0] -  allBasinFluxes[:, :, 0])

    mean_flow_precp = np.full((len(allBasinFluxes)),0)
    peak_flow_precp = np.full((len(allBasinFluxes)),0)
    mean_flow_T = np.full((len(allBasinFluxes)),0)
    peak_flow_T = np.full((len(allBasinFluxes)),0)
    Rain_runoff_ratio = np.full((len(allBasinFluxes)),0)
    annual_Q7_precp = np.full((len(allBasinFluxes)),0)
    annual_Q7_T = np.full((len(allBasinFluxes)), 0)
    for year in range(1995,2010):
        time_year = pd.date_range(f'{year}-10-01', f'{year+1}-09-30', freq='d')
        idx_start = test_span.get_loc(time_year[0])
        idx_end = test_span.get_loc(time_year[-1])+1
        annual_delta_precp = delta_precp[:, idx_start:idx_end]
        annual_delta_T = delta_T[:, idx_start:idx_end]
        annual_flow =  np.sum(allBasinFluxes[:, idx_start:idx_end,0],axis = 1)
        annual_flow_precp = np.sum(allBasinFluxes_precp[:, idx_start:idx_end, 0],axis = 1)
        annual_flow_T = np.sum(allBasinFluxes_T[:, idx_start:idx_end, 0],axis = 1)
        annual_obs = np.sum(test_obs[:, idx_start:idx_end], axis=1)

        peak_flow_precp = peak_flow_precp+  (np.max(allBasinFluxes_precp[:, idx_start:idx_end, 0],axis = -1)-np.max(allBasinFluxes[:, idx_start:idx_end, 0],axis = -1))/(np.max(allBasinFluxes[:, idx_start:idx_end, 0],axis = -1)+0.00001)
        peak_flow_T = peak_flow_T + (np.max(allBasinFluxes_T[:, idx_start:idx_end, 0],axis = -1)-np.max(allBasinFluxes[:, idx_start:idx_end, 0],axis = -1))/(np.max(allBasinFluxes[:, idx_start:idx_end, 0],axis = -1)+0.00001)


        # peak_flow_precp = peak_flow_precp+  annual_delta_precp[np.arange(annual_delta_precp.shape[0]), np.argmax(np.abs(annual_delta_precp), axis=-1)]/np.max(annual_flow,axis = -1)
        # peak_flow_T = peak_flow_T + annual_delta_T[
        #     np.arange(annual_delta_T.shape[0]), np.argmax(np.abs(annual_delta_T), axis=-1)]/np.max(annual_flow,axis = -1)
        mean_flow_precp = mean_flow_precp + (annual_flow_precp-annual_flow)/annual_flow
        mean_flow_T = mean_flow_T + (annual_flow_T-annual_flow)/annual_flow
        Rain_runoff_ratio = Rain_runoff_ratio +annual_obs/meanP

        Q7_precp = calc_Q7(allBasinFluxes_precp[:, idx_start:idx_end, 0])
        Q7_T = calc_Q7(allBasinFluxes_T[:, idx_start:idx_end, 0])
        Q7_ = calc_Q7(allBasinFluxes[:, idx_start:idx_end, 0])
        annual_Q7_precp = annual_Q7_precp + (Q7_precp-Q7_)/Q7_
        annual_Q7_T = annual_Q7_T +(Q7_T-Q7_)/Q7_
    peak_flow_precp = peak_flow_precp/len(range(1995,2010))
    peak_flow_T = peak_flow_T / len(range(1995, 2010))
    mean_flow_precp = mean_flow_precp/ len(range(1995, 2010))
    mean_flow_T = mean_flow_T / len(range(1995, 2010))
    annual_Q7_precp = annual_Q7_precp/len(range(1995, 2010))
    annual_Q7_T = annual_Q7_T / len(range(1995, 2010))
    Rain_runoff_ratio = Rain_runoff_ratio/ len(range(1995, 2010))
    mean_delta_flow_frac_area_precp[:,id_change] = mean_flow_precp
    mean_delta_flow_frac_area_T[:, id_change] = mean_flow_T

    peak_delta_flow_area_precp[:,id_change] = peak_flow_precp
    peak_delta_flow_area_T[:, id_change] = peak_flow_T

    delta_Q7_area_precp[:,id_change] = annual_Q7_precp
    delta_Q7_area_T[:, id_change] = annual_Q7_T



np.save(out+'/delta_Q7_area_precp.npy',delta_Q7_area_precp)
np.save(out+'/delta_Q7_area_T.npy',delta_Q7_area_T)
np.save(out+'/peak_delta_flow_area_precp.npy',peak_delta_flow_area_precp)
np.save(out+'/peak_delta_flow_area_T.npy',peak_delta_flow_area_T)


# aridity = attribute[:,np.where(np.array(attributeLst) == "aridity")[0][0]]


# level = [0.8,1.2,2.5]
# very_arid = np.where(aridity>level[2])[0]
# arid =np.where((aridity<=level[2]) & (aridity>level[1]))[0]
# humid =np.where((aridity<=level[1]) & (aridity>level[0]))[0]
# very_humid = np.where( (aridity<=level[0]))[0]

# basinsi_list = [very_arid,arid,humid,very_humid]
# climate_label = ['very_arid','arid','humid','very_humid']


# variables = [peak_delta_flow_area_precp,delta_Q7_area_precp, peak_delta_flow_area_T,delta_Q7_area_T]
# titles = [r'$\Delta P$',r'$\Delta P$',r'$\Delta T$',r'$\Delta T$']

# ylabel = [r'$\overline{\Delta Q_{max}/Q_{max}}$',r'$(\overline{(Q7_{perturbed} - Q7)/Q7})$',r'$\overline{\Delta Q_{max}/Q_{max})}$',r'$(\overline{(Q7_{perturbed} - Q7)/Q7})$']
# # Define the number of subplots based on the number of variables
# n_subplots = len(variables)
# fontsize = 24
# plt.rcParams.update({'font.size': fontsize})
# # Create a figure and a set of subplots
# fig, axes = plt.subplots(2, 2, figsize=(18, 16))  # Adjust figsize as needed
# # Define color labels

# color_labels = ['red','orange','green','darkblue','cyan','darkblue']
# ls_list=["--",":","dotted","dashdot"]
# # Loop over each variable and create a subplot
# for i, ax in enumerate(axes.flatten()):


#     # Plot the variable on the ith subplot
#     # Plot the variable on the ith subplot
#     for climatei in range(len(basinsi_list)):
#         color = color_labels[climatei]
#         # if climatei == 3 or climatei == 4:
#         #     for basinsi in basinsi_list[climatei]:
#         #         if basinsi in basinsi_list[climatei]:
#         #             color = color_labels[climatei]


#         #         ax.plot(np.sqrt(np.array(A_change_list)), variables[i][basinsi, :], ls='--', alpha=0.25, c=color)



#         median_line = np.nanpercentile(variables[i][basinsi_list[climatei],:],50,axis = 0)
#         upper_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],90,axis = 0)
#         lower_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],10,axis = 0)
#         ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.2)
#         ax.plot(np.sqrt(np.array(A_change_list)),median_line,lw = 2, ls = '-',c =color,label=climate_label[climatei])
#     # Set the title of the ith subplot
#     ax.set_title(titles[i])

#     # Add any other necessary plot customizations here
#     #ax.set_xlabel(r'sqrt(Area) (km)')
#     if i == 2 or i== 3 :
#         ax.set_xlabel(r'sqrt(Area) (km)')
#     ax.set_ylabel(ylabel[i])
#     if i == 3:
#         ax.set_ylim([-0.1,0])
#     if i == 1:
#         ax.set_ylim([0.01,0.03])
#     if i == 0:
#         ax.legend(loc='upper right',fontsize = 30)
#     ax.set_xlim([0,200])
# # Adjust the layout so that titles and labels don't overlap


# plt.tight_layout()
# plt.savefig("Partial_dependence_variables_forcing.png", dpi=300)
# plt.show(block =True)
# print("Done")