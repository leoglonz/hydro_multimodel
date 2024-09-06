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


def _basin_norm_to_m3_s(
    x: np.array, basin_area: np.array, to_norm: bool
) -> np.array:
    """
    Normalize or denormalize streamflow data with basin area and mean precipitation.

    The formula is as follows when normalizing (denormalize equation is its inversion):

    .. math:: normalized_x = \frac{x}{area * precipitation}

    Because units of streamflow, area, and precipitation are ft^3/s, km^2 and mm/day, respectively,
    and we need (m^3/day)/(m^3/day), we transform the equation as the code shows.

    Parameters
    ----------
    x
        data to be normalized or denormalized
    basin_area
        basins' area
    mean_prep
        basins' mean precipitation
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    nd = len(x.shape)
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    if x.shape[0] != basin_area.shape[0]:
        basin_area = np.expand_dims(basin_area,axis = 0)
        basin_area = np.tile(basin_area,(x.shape[0],1))
    temparea = np.tile(basin_area, (1, x.shape[1]))

    if to_norm is True:


        flow = (x  * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

    else:

        flow = (
            x
            * ((temparea * (10**6)) * (10 ** (-3)))
            / ( 3600 * 24)
        )

    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow




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



NSEloss_dataPred = pd.read_csv(  "/projects/mhpi/yxs275/model/scale_analysis/dPL_local_daymet_new_attr_NSE_loss_wo_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365/" +"/out0", dtype=np.float32, header=None).values

evaDict = [stat.statError(NSEloss_dataPred[:,-len(test_span):], test_obs)]
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
print("HBV model 'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))

NSE_NSE_loss_model = dataBox[0][0]

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

bias_pred,_= bias_meanflowratio_calc(NSEloss_dataPred[:,-len(test_span):], test_obs)


A_change_list = [5,10,20,30,50,80,100,200,300,500,800,1000,1200,1500,1800,2000,3000,4000,5000,6000,8000,10000,15000,20000,30000,40000]
prediction_out = f"/projects/mhpi/yxs275/model/scale_analysis/dPL_local_daymet_new_attr_NSE_loss_wo_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365/scaling_analysis"
if os.path.exists(prediction_out) is False:
    os.mkdir(prediction_out)
#filePathLst = [prediction_out+"/allBasinParaHBV_base_new",prediction_out+"/allBasinParaRout_base_new",prediction_out+"/allBasinFluxes_base_new"]

filePathLst = [prediction_out + f"/allBasinParaHBV_area_new_{A_change_list[0]}", prediction_out + f"/allBasinParaRout_area_new_{A_change_list[0]}",prediction_out+f"/allBasinFluxes_area_new_{A_change_list[0]}"]
allBasinParaHBV_base = np.load(filePathLst[0]+".npy")
allBasinParaRout_base = np.load(filePathLst[1]+".npy")
allBasinFluxes_base = np.load(filePathLst[2]+".npy")[:,-len(test_span):,:]


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




parameters = ["Beta","FC","K0","K1","K2","LP","PERC","UZL","TT","CFMAX","CFR","CWH","BETAET"]
selected_parameters = ["K0","K1","K2","Beta","FC","UZL"]



mean_flow_frac_area = np.full((len(attribute),len(A_change_list)),np.nan)
Q2_Q_area = np.full((len(attribute),len(A_change_list)),np.nan)
peak_flow_area = np.full((len(attribute),len(A_change_list)),np.nan)
Bias_area = np.full((len(attribute),len(A_change_list)),np.nan)

ET_area = np.full((len(attribute),len(A_change_list)),np.nan)
mean_flow_area = np.full((len(attribute),len(A_change_list)),np.nan)

HBVpara_area = np.full((len(attribute),len(A_change_list),len(selected_parameters)),np.nan)
Routpara_area = np.full((len(attribute),len(A_change_list),2),np.nan)

peak_flow_vol_area = np.full((len(attribute),len(A_change_list)),np.nan)






for id_change in range(len(A_change_list)):


    filePathLst = [prediction_out + f"/allBasinParaHBV_area_new_{A_change_list[id_change]}", prediction_out + f"/allBasinParaRout_area_new_{A_change_list[id_change]}",prediction_out+f"/allBasinFluxes_area_new_{A_change_list[id_change]}"]


    allBasinParaHBV = np.load(filePathLst[0]+".npy")
    allBasinParaRout = np.load(filePathLst[1]+".npy")
    allBasinFluxes = np.load(filePathLst[2] + ".npy")[:,-len(test_span):,:]

    Q2_Q = np.sum(allBasinFluxes[:,:,3],axis = 1)/(np.sum(allBasinFluxes[:,:,0],axis = 1)+0.00001)
    

    Bias,_ =bias_meanflowratio_calc(allBasinFluxes[:,:,0],test_obs)

    mean_flow = np.full((len(allBasinFluxes)),0)
    peak_flow = np.full((len(allBasinFluxes)),0)
    ET_flow = np.full((len(allBasinFluxes)),0)
    peak_flow_vol = np.full((len(allBasinFluxes)),0)

    for year in range(1995,2010):
        time_year = pd.date_range(f'{year}-10-01', f'{year+1}-09-30', freq='d')
        idx_start = test_span.get_loc(time_year[0])
        idx_end = test_span.get_loc(time_year[-1])

        peak_flow = peak_flow+np.max(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1)/(np.sum(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1)+0.00001)
        ET_flow = ET_flow + np.sum(allBasinFluxes[:,idx_start:idx_end+1,-1],axis = 1)
        
        mean_flow = mean_flow+np.sum(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1)
        peak_flow_vol = peak_flow_vol + _basin_norm_to_m3_s(np.max(allBasinFluxes[:,idx_start:idx_end+1,0],axis = 1,keepdims=True),basin_area=np.array(A_change_list[id_change:id_change+1]),to_norm=False)[:,0]

        
    _, mean_flow_ratio = bias_meanflowratio_calc(allBasinFluxes[:,:,0],allBasinFluxes_base[:,:,0])
    
    _, ET_ratio = bias_meanflowratio_calc(allBasinFluxes[:,:,-1],allBasinFluxes_base[:,:,-1])
    
    
    Q2_Q_area[:,id_change] = Q2_Q
    Bias_area[:,id_change] = Bias
    mean_flow_frac_area[:,id_change] = mean_flow_ratio

    mean_flow_area[:,id_change] = mean_flow/len(range(1995,2010))
    peak_flow_area[:,id_change] = peak_flow/len(range(1995,2010))
    
    ET_area[:,id_change] = ET_ratio
    peak_flow_vol_area[:,id_change] = peak_flow_vol/len(range(1995,2010))

    [C, Ind_para, SubInd_para] = np.intersect1d(selected_parameters, parameters, return_indices=True)
    HBVpara_area[:,id_change,:] = allBasinParaHBV[:, np.sort(SubInd_para)]
    Routpara_area[:,id_change, :] = allBasinParaRout



aridity = attribute[:,np.where(np.array(attributeLst) == "aridity")[0][0]]


level = [0.75,1.2,2] #aridity
# level = [500,1000,2000]  #elevation
#level = [0.75,1.2,2] #aridity
very_arid = np.where(aridity>level[2])[0]
arid =np.where((aridity<=level[2]) & (aridity>level[1]))[0]
humid =np.where((aridity<=level[1]) & (aridity>level[0]))[0]
very_humid = np.where( (aridity<=level[0]))[0]

basinsi_list = [very_arid,arid,humid,very_humid]
climate_label = ['arid','semi-arid','semi-humid','humid']
# climate_label = ['high elevation','high-mid elevation','low-mid elevation','low elevation']

variables = [mean_flow_frac_area, Q2_Q_area,ET_area , peak_flow_area]
titles = ['(a) Water supply','(b) Baseflow ratio','(c) Evapotranspiration','(d) Peak flow ratio']
ylabels = [r'$Q$/$Q_{minA}}$', '$Q_{base}$/$Q$',r'$ET$/$ET_{minA}$',r'$Q_{max}/Q_{annual}$']
# Define the number of subplots based on the number of variables
n_subplots = len(variables)
fontsize = 30
plt.rcParams.update({'font.size': fontsize})
# Create a figure and a set of subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 16))  # Adjust figsize as needed
# Define color labels

color_labels = ['orangered','orange','darkturquoise','darkcyan','cyan','darkblue']
ls_list=["solid","dashed","dotted","dashdot"]
# Loop over each variable and create a subplot
for i, ax in enumerate(axes.flatten()):


    # Plot the variable on the ith subplot
    # Plot the variable on the ith subplot
    if i<len(variables):
        for climatei in range(len(basinsi_list)):
            color = color_labels[climatei]
            # if climatei == 3 or climatei == 4:
            #     for basinsi in basinsi_list[climatei]:
            #         if basinsi in basinsi_list[climatei]:
            #             color = color_labels[climatei]


            #         ax.plot(np.sqrt(np.array(A_change_list)), variables[i][basinsi, :], ls='--', alpha=0.25, c=color)



            median_line = np.nanpercentile(variables[i][basinsi_list[climatei],:],50,axis = 0)
            upper_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],75,axis = 0)
            lower_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],25,axis = 0)
            ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.15)
            ax.plot(np.sqrt(np.array(A_change_list)),median_line,lw = 5, ls = ls_list[climatei],c =color,label=climate_label[climatei])
        # Set the title of the ith subplot
        ax.set_title(titles[i],pad = 20)

        # Add any other necessary plot customizations here
        #ax.set_xlabel(r'sqrt(Area) (km)')
        if i == 2 or i== 3:
            ax.set_xlabel(r'$\sqrt{A_c}$ (km)')
        ax.set_ylabel(ylabels[i])

        if i == 0:
            ax.set_ylim([0.7,1.3])
        if i == 2:
            ax.set_ylim([0.9,1.1])
        
        if i == 3:
            ax.legend(loc='upper right',fontsize = 30,frameon = False)
        ax.set_xlim([0,200])
# Adjust the layout so that titles and labels don't overlap

# plt.title("NSE loss model")
plt.tight_layout()
plt.savefig("Partial_dependence_variables.png", dpi=100)

print("Done")






variables = [Bias_area,peak_flow_vol_area]
titles = ['(a) Water deficit','(b) Annual maximum discharge']
ylabels = [ r'Relative total bias','$Q_{max} (m^3/s)$']
# Define the number of subplots based on the number of variables

n_subplots = len(variables)
fontsize = 30
plt.rcParams.update({'font.size': fontsize})
# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # Adjust figsize as needed
# Define color labels

color_labels = ['orangered','orange','darkturquoise','darkcyan','cyan','darkblue']
ls_list=["solid","dashed","dotted","dashdot"]
# Loop over each variable and create a subplot
for i, ax in enumerate(axes.flatten()):


    # Plot the variable on the ith subplot
    # Plot the variable on the ith subplot
    if i<len(variables):
        for climatei in range(len(basinsi_list)):
            color = color_labels[climatei]
            # if climatei == 3 or climatei == 4:
            #     for basinsi in basinsi_list[climatei]:
            #         if basinsi in basinsi_list[climatei]:
            #             color = color_labels[climatei]


            #         ax.plot(np.sqrt(np.array(A_change_list)), variables[i][basinsi, :], ls='--', alpha=0.25, c=color)



            median_line = np.nanpercentile(variables[i][basinsi_list[climatei],:],50,axis = 0)
            upper_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],75,axis = 0)
            lower_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],25,axis = 0)
            ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.15)
            ax.plot(np.sqrt(np.array(A_change_list)),median_line,lw = 5, ls = ls_list[climatei],c =color,label=climate_label[climatei])
        # Set the title of the ith subplot
        ax.set_title(titles[i],pad = 20)

        # Add any other necessary plot customizations here
        #ax.set_xlabel(r'sqrt(Area) (km)')
        if i == 0 or i== 1:
            ax.set_xlabel(r'$\sqrt{A_c}$ (km)')
        ax.set_ylabel(ylabels[i])
        if i == 0:
            ax.set_ylim([-0.2,0.2])
        
        if i == 1:
            ax.legend(loc='upper left',fontsize = 30,frameon = False)
        ax.set_xlim([0,200])
# Adjust the layout so that titles and labels don't overlap

# plt.title("NSE loss model")
plt.tight_layout()
plt.savefig("Partial_dependence_variables_S.png", dpi=100)

print("Done")


















delta_Q7_area_precp = np.load('delta_Q7_area_precp.npy')
delta_Q7_area_T = np.load('delta_Q7_area_T.npy')
peak_delta_flow_area_precp= np.load('peak_delta_flow_area_precp.npy')
peak_delta_flow_area_T = np.load('peak_delta_flow_area_T.npy')


#climate_label = ['arid','semi-arid','semi-humid','humid']
variables = [peak_delta_flow_area_precp,delta_Q7_area_T]
titles = ['(a) Flood sensitivity \n to +1% precipitation','(b) Low flow sensitivity \n to +1$^{\circ}$ temperature']

ylabel = [r'$\Delta Q_{max}/Q_{max}$',r'$\Delta Q_{7}/Q_{7}$']
# Define the number of subplots based on the number of variables
n_subplots = len(variables)
fontsize = 28
plt.rcParams.update({'font.size': fontsize})
# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # Adjust figsize as needed
# Define color labels

color_labels = ['orangered','orange','darkturquoise','darkcyan','cyan','darkblue']
ls_list=["solid","dashed","dotted","dashdot"]
# Loop over each variable and create a subplot
for i, ax in enumerate(axes.flatten()):


    # Plot the variable on the ith subplot
    # Plot the variable on the ith subplot
    for climatei in range(len(basinsi_list)):
        color = color_labels[climatei]
        # if climatei == 3 or climatei == 4:
        #     for basinsi in basinsi_list[climatei]:
        #         if basinsi in basinsi_list[climatei]:
        #             color = color_labels[climatei]


        #         ax.plot(np.sqrt(np.array(A_change_list)), variables[i][basinsi, :], ls='--', alpha=0.25, c=color)



        median_line = np.nanpercentile(variables[i][basinsi_list[climatei],:],50,axis = 0)
        upper_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],75,axis = 0)
        lower_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],25,axis = 0)
        ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.15)
        ax.plot(np.sqrt(np.array(A_change_list)),median_line,lw = 5, ls = ls_list[climatei],c =color,label=climate_label[climatei])
    # Set the title of the ith subplot
    ax.set_title(titles[i], pad=20)

    # Add any other necessary plot customizations here
    #ax.set_xlabel(r'sqrt(Area) (km)')
    if i == 0 or i== 1 :
        ax.set_xlabel(r'$\sqrt{A_c}$ (km)')
    ax.set_ylabel(ylabel[i])
    if i == 1:
        ax.set_ylim([-0.1,0.1])
    if i == 1:
        ax.legend(loc='upper right',fontsize = 30,frameon = False)
    ax.set_xlim([0,200])
# Adjust the layout so that titles and labels don't overlap


plt.tight_layout()
plt.savefig("Partial_dependence_variables_forcing.png", dpi=300)
plt.show(block =True)
print("Done")






variables = [delta_Q7_area_precp, peak_delta_flow_area_T]
titles = ['(a) Low flow sensitivity \n to +1% precipitation','(b) Flood sensitivity \n to +1$^{\circ}$ temperature']

ylabel = [r'$\Delta Q_{7}/Q_{7}$',r'$\Delta Q_{max}/Q_{max})$']
# Define the number of subplots based on the number of variables
n_subplots = len(variables)
fontsize = 28
plt.rcParams.update({'font.size': fontsize})
# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # Adjust figsize as needed
# Define color labels

color_labels = ['orangered','orange','darkturquoise','darkcyan','cyan','darkblue']
ls_list=["solid","dashed","dotted","dashdot"]
# Loop over each variable and create a subplot
for i, ax in enumerate(axes.flatten()):


    # Plot the variable on the ith subplot
    # Plot the variable on the ith subplot
    for climatei in range(len(basinsi_list)):
        color = color_labels[climatei]
        # if climatei == 3 or climatei == 4:
        #     for basinsi in basinsi_list[climatei]:
        #         if basinsi in basinsi_list[climatei]:
        #             color = color_labels[climatei]


        #         ax.plot(np.sqrt(np.array(A_change_list)), variables[i][basinsi, :], ls='--', alpha=0.25, c=color)



        median_line = np.nanpercentile(variables[i][basinsi_list[climatei],:],50,axis = 0)
        upper_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],75,axis = 0)
        lower_line  = np.nanpercentile(variables[i][basinsi_list[climatei],:],25,axis = 0)
        ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.15)
        ax.plot(np.sqrt(np.array(A_change_list)),median_line,lw = 5, ls = ls_list[climatei],c =color,label=climate_label[climatei])
    # Set the title of the ith subplot
    ax.set_title(titles[i], pad=20)

    # Add any other necessary plot customizations here
    #ax.set_xlabel(r'sqrt(Area) (km)')
    if i == 0 or i== 1 :
        ax.set_xlabel(r'$\sqrt{A_c}$ (km)')
    ax.set_ylabel(ylabel[i])
    if i == 0:
        ax.set_ylim([0.005,0.05])
    if i == 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1.05),fontsize = 30,frameon = False)
    ax.set_xlim([0,200])
# Adjust the layout so that titles and labels don't overlap


plt.tight_layout()
plt.savefig("Partial_dependence_variables_forcing_S.png", dpi=300)
plt.show(block =True)
print("Done")












variables = [mean_flow_frac_area, Q2_Q_area,peak_delta_flow_area_precp, delta_Q7_area_T,]
titles = ['(a) ', '(b)', '(c)', '(d)', '(e)', '(f)','(g)']
ylabels = ['Scale change ratio of\nwater supply  $R$($Q$/$Q_{minA}}$)', 'Scale change ratio of\ngroundwater contribution $R$($Q_{base}$/$Q$)','Scale change ratio of\npeak flow $R$($\Delta Q_{max}/Q_{max}$)\nresponse to +1% precipitation','Scale change ratio of\ndrought metric $R$($\Delta Q_{7}/Q_{7}$)\nresponse to +1$^{\circ}$ temperature ']
titles = [titles[titleI] + ' ' + ylabels[titleI] for titleI in range(len(ylabels))]
plt.rcParams.update({'font.size': 24})

fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # Adjust figsize as needed

for i, ax in enumerate(axes.flatten()):
    m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-65, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x, y = m(lon, lat)

    max_element_indices = np.argmax(np.abs(variables[i]-1), axis=1)

    # Extract the first element and the absolute maximum element in each row
    first_elements = variables[i][np.arange(variables[i].shape[0]), 0]  # First elements of each row
    last_elements = variables[i][np.arange(variables[i].shape[0]), -1]
    #max_elements = variables[i][np.arange(variables[i].shape[0]), max_element_indices]  # Absolute maximum elements of each row


    delta_variable = (last_elements - first_elements) / last_elements              #last_elements

    print(len(delta_variable))
    vmin = np.nanpercentile(delta_variable,10)
    vmax = np.nanpercentile(delta_variable,90)
    vvmax = max(abs(vmin),abs(vmax))
    scatter = ax.scatter(x, y, s=50, c=delta_variable, cmap=plt.cm.seismic, vmin=-vvmax, vmax=vvmax)
    ax.set_title(titles[i])
    cbar = plt.colorbar(scatter, ax=ax, fraction = 0.11, location='bottom', pad=0.05)

   # cbar.formatter.set_powerlimits((0, 0))



plt.tight_layout()
plt.savefig("Variable_change_map.png", dpi=300)
print("Done")


variables = [mean_flow_frac_area, Q2_Q_area]
titles = ['(a) ', '(b)', '(c)', '(d)', '(e)', '(f)','(g)']
ylabels = ['Scale change ratio of\nwater supply  $R$($Q$/$Q_{minA}}$)', 'Scale change ratio of\ngroundwater contribution $R$($Q_{base}$/$Q$)']
titles = [titles[titleI] + ' ' + ylabels[titleI] for titleI in range(len(ylabels))]
plt.rcParams.update({'font.size': 24})

fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Adjust figsize as needed

for i, ax in enumerate(axes.flatten()):
    m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-65, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x, y = m(lon, lat)

    max_element_indices = np.argmax(np.abs(variables[i]-1), axis=1)

    # Extract the first element and the absolute maximum element in each row
    first_elements = variables[i][np.arange(variables[i].shape[0]), 0]  # First elements of each row
    last_elements = variables[i][np.arange(variables[i].shape[0]), -1]
    #max_elements = variables[i][np.arange(variables[i].shape[0]), max_element_indices]  # Absolute maximum elements of each row


    delta_variable = (last_elements - first_elements) / last_elements              #last_elements

    print(len(delta_variable))
    vmin = np.nanpercentile(delta_variable,10)
    vmax = np.nanpercentile(delta_variable,90)
    vvmax = max(abs(vmin),abs(vmax))
    scatter = ax.scatter(x, y, s=50, c=delta_variable, cmap=plt.cm.seismic, vmin=-vvmax, vmax=vvmax)
    ax.set_title(titles[i])
    cbar = plt.colorbar(scatter, ax=ax, fraction = 0.11, location='bottom', pad=0.05)

   # cbar.formatter.set_powerlimits((0, 0))



plt.tight_layout()
plt.savefig("Variable_change_map.png", dpi=300)
print("Done")















variables = [delta_Q7_area_precp,peak_delta_flow_area_T, bias_pred]
titles = ['(a) ', '(b)', '(c)', '(d)', '(e)', '(f)','(g)']
ylabels = ['Scale change ratio of\ndrought metric $R$($\Delta Q_{7}/Q_{7}$)\nresponse to +1% precipitation','Scale change ratio of\npeak flow $R$($\Delta Q_{max}/Q_{max}$)\nresponse to +1$^{\circ}$ temperature',]
titles = [titles[titleI] + ' ' + ylabels[titleI] for titleI in range(len(ylabels))]
plt.rcParams.update({'font.size': 24})

fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Adjust figsize as needed

for i, ax in enumerate(axes.flatten()):
    m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-65, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x, y = m(lon, lat)

    max_element_indices = np.argmax(np.abs(variables[i]-1), axis=1)

    # Extract the first element and the absolute maximum element in each row
    first_elements = variables[i][np.arange(variables[i].shape[0]), 0]  # First elements of each row
    last_elements = variables[i][np.arange(variables[i].shape[0]), -1]
    #max_elements = variables[i][np.arange(variables[i].shape[0]), max_element_indices]  # Absolute maximum elements of each row


    delta_variable = (last_elements - first_elements) / last_elements              #last_elements

    print(len(delta_variable))
    vmin = np.nanpercentile(delta_variable,10)
    vmax = np.nanpercentile(delta_variable,90)
    vvmax = max(abs(vmin),abs(vmax))
    scatter = ax.scatter(x, y, s=50, c=delta_variable, cmap=plt.cm.seismic, vmin=-vvmax, vmax=vvmax)
    ax.set_title(titles[i])
    cbar = plt.colorbar(scatter, ax=ax, fraction = 0.11, location='bottom', pad=0.05)

   # cbar.formatter.set_powerlimits((0, 0))



plt.tight_layout()
plt.savefig("Variable_change_map_S.png", dpi=300)
print("Done")









import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors
import numpy as np

# Set the font size for better readability
plt.rcParams.update({'font.size': 24})

# Define the boundaries and colors
boundaries = [0, 0.75,1.2, 2, 7]
colors = ['darkblue','green','orange','red']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
m = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-65, resolution='i', ax=ax)
m.drawcoastlines()
m.drawcountries()
m.drawstates()
x, y = m(lon, lat)
scatter = ax.scatter(x, y, s=30, c=aridity, cmap=cmap, norm=norm)

ax.set_title('Aridity')
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, ticks=boundaries[:-1])
# Add legend for each color
labels = ['humid','semi-humid', 'semi-arid','arid']
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in colors]
legend = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))



plt.tight_layout()
plt.savefig("aridity_map.png", dpi=300)
print("Done")



titles = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
ylabels = np.array(parameters)[np.sort(SubInd_para)]

# Define the number of subplots based on the number of variables
n_subplots = HBVpara_area.shape[-1]

fontsize = 30
plt.rcParams.update({'font.size': fontsize})

# Create a figure and a set of subplots
fig, axes = plt.subplots(2, 3, figsize=(25, 15))  # Adjust figsize as needed


color_labels = ['orangered','orange','darkturquoise','darkcyan','cyan','darkblue']
ls_list=["solid","dashed","dotted","dashdot"]
# Loop over each variable and create a subplot
for i, ax in enumerate(axes.flatten()):

    # Plot the variable on the ith subplot
    # Plot the variable on the ith subplot
    if i<n_subplots:
        for climatei in range(len(basinsi_list)):

            color = color_labels[climatei]

            median_line = np.nanpercentile(HBVpara_area[basinsi_list[climatei],:,i],50,axis = 0)
            upper_line  = np.nanpercentile(HBVpara_area[basinsi_list[climatei],:,i],90,axis = 0)
            lower_line  = np.nanpercentile(HBVpara_area[basinsi_list[climatei],:,i],10,axis = 0)
            ax.fill_between(np.sqrt(np.array(A_change_list)), lower_line,upper_line, color=color, alpha=0.15)
            ax.plot(np.sqrt(np.array(A_change_list)),median_line,lw = 5, ls = ls_list[climatei],c =color,label=climate_label[climatei])
    # Set the title of the ith subplot


        # Set the title of the ith subplot
        ax.set_title(titles[i])

        # Add any other necessary plot customizations here
        ax.set_xlabel(r'$\sqrt{A_c}$ (km)')
        ax.set_ylabel(ylabels[i])

    if i == 5:
        ax.legend(loc='lower right',fontsize = 30,frameon = False)
# Adjust the layout so that titles and labels don't overlap
plt.tight_layout()

plt.savefig("Partial_dependence_parameters.png", dpi=300)
print("Done")



RMSEloss_model_path  = '/projects/mhpi/yxs275/model/scale_analysis/dPL_local_daymet_new_attr_RMSE_loss_w_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365/'
RMSEloss_model_dataPred = pd.read_csv(  RMSEloss_model_path+"/out0", dtype=np.float32, header=None).values



bias_RMSEloss_model_dataPred,_= bias_meanflowratio_calc(RMSEloss_model_dataPred[:,-len(test_span):], test_obs)
evaDict = [stat.statError(RMSEloss_model_dataPred[:,-len(test_span):], test_obs)]
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
print("HBV model 'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))
NSE_RMSEloss_model = dataBox[0][0]


RMSEloss_wo_log_model_path  = '/projects/mhpi/yxs275/model/scale_analysis/dPL_local_daymet_new_attr_RMSE_loss_wo_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365/'
RMSEloss_wo_log_model_dataPred = pd.read_csv(  RMSEloss_wo_log_model_path+"/out0", dtype=np.float32, header=None).values

NSEloss_w_log_model_path  = '/projects/mhpi/yxs275/model/scale_analysis/dPL_local_daymet_new_attr_NSE_loss_w_log_with_Cr/exp_EPOCH100_BS100_RHO365_HS512_trainBuff365/'
NSEloss_w_log_model_dataPred = pd.read_csv(  NSEloss_w_log_model_path+"/out0", dtype=np.float32, header=None).values


average_simulation = (RMSEloss_model_dataPred+NSEloss_dataPred+RMSEloss_wo_log_model_dataPred+ NSEloss_w_log_model_dataPred)/4.0
bias_average_model_dataPred,_= bias_meanflowratio_calc(average_simulation[:,-len(test_span):], test_obs)

evaDict = [stat.statError(average_simulation[:,-len(test_span):], test_obs)]
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
print("HBV model 'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))
NSE_average_model = dataBox[0][0]



##Plotting

nbin = 4
lower_bound = 0
upper_bound = 24000
#bins = np.linspace(lower_bound, upper_bound, nbin + 1)
bin_length = (upper_bound - lower_bound) / (nbin-1)
bins =np.array([0,8000,16000,24000,32000])
#bins_split =np.array([0,1000,2000,5000,10000,50000])
bins_split =np.array([0,0.75,1.2,2,7])

#area_selected =basin_area[:,0]
area_selected =aridity
area_bin_index = np.digitize(area_selected, bins_split)
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
labels = []
for bin_i in range(len(bins)-1):
    labels.append(f'{bins_split[bin_i]}~{bins_split[bin_i+1]}')

plot1 = ax.boxplot( [ NSE_NSE_loss_model[np.where(area_bin_index == i)][~np.isnan(NSE_NSE_loss_model[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+1*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="pink", color="k") ,widths = bin_length/6)
plot2 = ax.boxplot( [ NSE_RMSEloss_model[np.where(area_bin_index == i)][~np.isnan(NSE_RMSEloss_model[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+2*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="lightblue", color="k") ,widths = bin_length/6)
plot3 = ax.boxplot( [ NSE_average_model[np.where(area_bin_index == i)][~np.isnan(NSE_average_model[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+3*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="orange", color="k") ,widths = bin_length/6)

for whisker in plot1['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot1['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot1['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot1['medians']:
    median.set(ls='-', linewidth=2,color = "k")

for whisker in plot2['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot2['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot2['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot2['medians']:
    median.set(ls='-', linewidth=2,color = "k")

for whisker in plot3['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot3['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot3['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot3['medians']:
    median.set(ls='-', linewidth=2,color = "k")

y_upper = 1.1
y_lower = -0.4
yrange = y_upper-y_lower
for i in range(1,nbin+1):

    num = len(area_selected[np.where(area_bin_index == i)])
    ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.08*(y_upper-y_lower), f'{num} sites')



ax.add_patch( Rectangle(( 700, y_lower+0.2*yrange), 200, yrange*0.05,  fc = "pink",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.2*yrange, r"$\delta$HBV NSE loss")

ax.add_patch( Rectangle(( 700, y_lower+0.1*yrange), 200, yrange*0.05,  fc = "lightblue",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.1*yrange, r"$\delta$HBV RMSE loss")
ax.add_patch( Rectangle(( 700, y_lower+0*yrange), 200, yrange*0.05,  fc = "orange",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0*yrange, r"$\delta$HBV averaged simulation")



ax.set_ylabel("NSE")
ax.set_xlabel(r"Aridity")
#ax.set_xlabel(r"Drainage area (km$^2$)")

ax.set_yticks(np.arange(y_lower,y_upper,0.2))
ax.set_ylim([y_lower,y_upper])
ax.set_xlim([lower_bound,upper_bound+bin_length])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.vlines([-0.5,0,0.5], -2, 4,color ="k",linestyles='--',lw = 1.5)
ax.hlines([0], 0, 100000,color ="k",linestyles='--',lw = 1.5)
ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), y_lower,y_upper,color ="k",linestyles='--',lw = 2.5)
tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
ax.set_xticks(tick_positions)
#ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
ax.set_xticklabels(labels)

plt.savefig("boxplot_NSE_aridity.png", dpi=300)
plt.show(block=True)

print("Done")





nbin = 4
lower_bound = 0
upper_bound = 24000
#bins = np.linspace(lower_bound, upper_bound, nbin + 1)
bin_length = (upper_bound - lower_bound) / (nbin-1)
bins =np.array([0,8000,16000,24000,32000])
#bins_split =np.array([0,1000,2000,5000,10000,50000])
bins_split =np.array([0,0.75,1.2,2,7])

#area_selected =basin_area[:,0]
area_selected =aridity
area_bin_index = np.digitize(area_selected, bins_split)
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
labels = []
for bin_i in range(len(bins)-1):
    labels.append(f'{bins_split[bin_i]}~{bins_split[bin_i+1]}')

plot1 = ax.boxplot( [ bias_pred[np.where(area_bin_index == i)][~np.isnan(bias_pred[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+1*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="pink", color="k") ,widths = bin_length/6)
plot2 = ax.boxplot( [ bias_RMSEloss_model_dataPred[np.where(area_bin_index == i)][~np.isnan(bias_RMSEloss_model_dataPred[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+2*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="lightblue", color="k") ,widths = bin_length/6)
plot3 = ax.boxplot( [ bias_average_model_dataPred[np.where(area_bin_index == i)][~np.isnan(bias_average_model_dataPred[np.where(area_bin_index == i)])] for i in range(1,nbin+1) ], vert=True,showfliers=False, positions=bins[:-1]+3*bin_length/4.0,patch_artist=True,boxprops=dict(facecolor="orange", color="k") ,widths = bin_length/6)






for whisker in plot1['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot1['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot1['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot1['medians']:
    median.set(ls='-', linewidth=2,color = "k")

for whisker in plot2['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot2['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot2['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot2['medians']:
    median.set(ls='-', linewidth=2,color = "k")
for whisker in plot3['whiskers']:
    whisker.set(ls='-', linewidth=2,color = "k")
for cap in plot3['caps']:
    cap.set(ls='-', linewidth=2,color = "k")
for box in plot3['boxes']:
    box.set(ls='-', linewidth=2)
for median in plot3['medians']:
    median.set(ls='-', linewidth=2,color = "k")

y_upper = 1.1
y_lower = -1
yrange = y_upper-y_lower
for i in range(1,nbin+1):

    num = len(area_selected[np.where(area_bin_index == i)])
    ax.text(bin_length/4.0+(i-1)*bin_length+lower_bound,y_upper-0.08*(y_upper-y_lower), f'{num} sites')


ax.add_patch( Rectangle(( 700, y_lower+0.2*yrange), 200, yrange*0.05,  fc = "pink",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.2*yrange, r"$\delta$HBV NSE loss")

ax.add_patch( Rectangle(( 700, y_lower+0.1*yrange), 200, yrange*0.05,  fc = "lightblue",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0.1*yrange, r"$\delta$HBV RMSE loss")
ax.add_patch( Rectangle(( 700, y_lower+0*yrange), 200, yrange*0.05,  fc = "orange",  ec ='k',ls = "--" , lw = 2) )
ax.text(1000, y_lower+0*yrange, r"$\delta$HBV averaged simulation")

ax.set_ylabel("Total bias ratio")
ax.set_xlabel(r"Aridity")
#ax.set_xlabel(r"Drainage area (km$^2$)")

ax.set_yticks(np.arange(y_lower,y_upper,0.2))
ax.set_ylim([y_lower,y_upper])
ax.set_xlim([lower_bound,upper_bound+bin_length])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# ax.vlines([-0.5,0,0.5], -2, 4,color ="k",linestyles='--',lw = 1.5)
ax.hlines([0], 0, 100000,color ="k",linestyles='--',lw = 1.5)
ax.vlines(np.arange(lower_bound+bin_length,upper_bound+bin_length,bin_length), y_lower,y_upper,color ="k",linestyles='--',lw = 2.5)
tick_positions = np.arange(lower_bound, upper_bound+bin_length, bin_length) + bin_length / 2
ax.set_xticks(tick_positions)
#ax.set_xticks(np.arange(lower_bound,upper_bound+bin_length,bin_length)+bin_length/2,labels)
ax.set_xticklabels(labels)

plt.savefig("boxplot_bias_aridity.png", dpi=300)
plt.show(block=True)

print("Done")







