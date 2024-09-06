import numpy as np
import pandas as pd

import json
import glob

#data_split_folder = "/data/yxs275/CONUS_data/HUC10/dPL_version2/Qr_for_each_basins/"
# smallBasinList = "/data/yxs275/CONUS_data/HUC10/smallerBasin/traningbasin_GSCD.xlsx"
# smallBasin = pd.read_excel(smallBasinList).values
# smallBasinID = []
# for item in smallBasin:
#     if (item[0].find("USGS")!=-1 ):
#         smallBasinID.append(item[0].split("'")[1].split("USGS")[-1])


data_folder_3200 = "/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/gages/dataCONUS3200/"


with open(data_folder_3200+'train_data_dict.json') as f:
    train_data_dict = json.load(f)
smallBasinID = train_data_dict['sites_id']




data_folder = "/data/yxs275/CONUS_data/HUC10/version_1_11_2014_continental_routing/"
attributeALL_df = pd.read_csv(data_folder + "attributes.csv")
basinID = attributeALL_df.gage_ID.values
HUC10_area = attributeALL_df.area.values

max_HUC10_area = 1000

GAGEII_folder = "/data/yxs275/CONUS_data/all_GAGEII/gages/dataGAGEall/"
GAGEII_flow =np.load(GAGEII_folder+"train_flow.npy")
GAGEII_attr = np.load(GAGEII_folder+"train_attr.npy")

with open(GAGEII_folder+'train_data_dict.json') as f:
    train_data_dict = json.load(f)

attributeAllLst  = train_data_dict['constant_cols']
GAGEIIAreaName = "DRAIN_SQKM"
GAGEII_area = GAGEII_attr[:,np.where(np.array(attributeAllLst)=="DRAIN_SQKM")[0]]

GAGEII_ID = train_data_dict['sites_id']

intercepted_folder = "/data/yxs275/CONUS_data/GAGEII_in_HUC10_v3/"
GAGEII_in_HUC10_files = glob.glob(intercepted_folder+"*.csv")
GAGEII_in_HUC10_files.sort()
selected_GAGE = []
selected_Basin = []
selected_gageIdx = []
selected_huc10Idx = []
GAGEII_ID_Int = np.array(GAGEII_ID).astype(int)
smallBasinID_Int = np.array(smallBasinID).astype(int)
for file in GAGEII_in_HUC10_files:
    basinI = file.split("/")[-1].split(".")[0]
    try:
        basinIdx = np.where(np.array(basinID)==int(basinI))[0][0]
    except:
        continue
    basinI_area = HUC10_area[basinIdx]

    gage_df = pd.read_csv(file)
    gageIDsIn = gage_df.Gage.values
    gageIDsIn_percentage = gage_df.Percentages.values
    gageIDsIn = gageIDsIn[np.where(gageIDsIn_percentage>0.6)]
    gageIDsIn_percentage = gageIDsIn_percentage[np.where(gageIDsIn_percentage > 0.6)]
    gageI_areaList  = []
    gageIdx_List = []
    IsSmallBasin = False
    for gageI in gageIDsIn:

        # if gageI in smallBasinID_Int:
        IsSmallBasin = True

        gageIdx = np.where(GAGEII_ID_Int == gageI)[0][0]


        gageI_area = GAGEII_area[gageIdx,0]
        gageI_areaList.append(gageI_area)
        gageIdx_List.append(gageIdx)
    if IsSmallBasin:


        closest_index = min(range(len(gageI_areaList)), key=lambda i: abs(gageI_areaList[i] - basinI_area))
        decide_gageIDIn = GAGEII_ID[gageIdx_List[closest_index]]

        if (gageI_areaList[closest_index]/basinI_area>=0.6 and gageI_areaList[closest_index]/basinI_area<=1.4):
            print("Selected", len(gageI_areaList), " basin", "Basin ", basinI, "area, ", basinI_area,
                  " is comparing to ", decide_gageIDIn, "area, ", gageI_areaList[closest_index])

            if gageI not in smallBasinID_Int:
                print(decide_gageIDIn, "is not trained")

            selected_Basin.append(basinI)
            selected_GAGE.append(decide_gageIDIn)
            # selected_gageIdx.append(np.where(np.array(smallBasinID) == decide_gageIDIn)[0][0])
            # selected_huc10Idx.append(basinIdx)
print('Selected ',len(selected_GAGE), "Gages")
np.save(data_folder+"selected_Basin_matched_v3.npy",np.array(selected_Basin))
np.save(data_folder+"selected_GAGE_matched_v3.npy",np.array(selected_GAGE))

# np.save(data_folder+"selected_gageIdx.npy",np.array(selected_gageIdx))
# np.save(data_folder+"selected_huc10Idx.npy",np.array(selected_huc10Idx))

# Qr_folder = "/data/yxs275/CONUS_data/HUC10/dPL_version3_12_5/exp_EPOCH50_BS100_RHO365_HS256_trainBuff365/"
# data_folder = "/data/yxs275/CONUS_data/HUC10/version_2_11_25/"
# attributeALL_df = pd.read_csv(data_folder + "attributes.csv")
# basinID = attributeALL_df.gage_ID.values
# batchSize = 1000
# iS = np.arange(0, len(basinID), batchSize)
# iE = np.append(iS[1:], len(basinID))
# for item in range(len(iS)):
#     # Qr_Batch = np.load(data_folder + f"forcings_{iS[item]}_{iE[item]}.npy")
#     Qr_Batch = pd.read_csv( Qr_folder + f"Qr_{iS[item]}_{iE[item]}", dtype=np.float32, header=None).values
#     attributeBatch_file = data_folder+f"attributes_{iS[item]}_{iE[item]}.csv"
#     attributeBatch_df = pd.read_csv(attributeBatch_file)
#     attributeBatch_ID = attributeBatch_df.gage_ID.values
#     for idx, ID in enumerate(attributeBatch_ID):
#         ID_str = str(ID).zfill(10)
#         Qr = Qr_Batch[idx:idx+1,:]
#         np.save(Qr_folder+"basin_split/"+f"{ID_str}.npy",Qr)


