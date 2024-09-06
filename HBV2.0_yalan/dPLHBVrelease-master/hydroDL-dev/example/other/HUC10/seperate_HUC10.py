import numpy as np
import pandas as pd
import os
import json
import tqdm
## Use this for separating HUC10s
# # Load your shapefile
data_folder = "/data/yxs275/CONUS_data/HUC10/version_1_11_2014_continental_routing/"
Qr_folder = "/data/yxs275/CONUS_data/HUC10/LSTM_local_daymet_filled_withNaN_NSE/"
data_split_folder = Qr_folder +"/basin_split/"
if os.path.exists(data_split_folder) is False:
    os.mkdir(data_split_folder)
attributeALL_df = pd.read_csv(data_folder + "attributes.csv")
basinID = attributeALL_df.gage_ID.values
batchSize = 1000
iS = np.arange(0, len(basinID), batchSize)
iE = np.append(iS[1:], len(basinID))
for item in range(len(iS)):

    Qr_Batch = pd.read_csv(Qr_folder + f"Qr_{iS[item]}_{iE[item]}", dtype=np.float32, header=None).values
    attributeBatch_file = data_folder+f"attributes_{iS[item]}_{iE[item]}.csv"
    attributeBatch_df = pd.read_csv(attributeBatch_file)
    attributeBatch_ID = attributeBatch_df.gage_ID.values
    for idx, ID in enumerate(attributeBatch_ID):
        ID_str = str(int(ID)).zfill(10)
        Qr = Qr_Batch[idx:idx+1,:]
        np.save(data_split_folder+f"{ID_str}.npy",Qr)

#
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# from tqdm import tqdm, trange
# import zarr
#
#
# data_folder = Path(
#     "/data/yxs275/CONUS_data/HUC10/version_1_11_2014_continental_routing/"
# )
# qr_folder = Path(
#     "/data/yxs275/CONUS_data/HUC10/dPL_1_11_2024/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/"
# )
# data_split_path = Path(
#     "/data/yxs275/CONUS_data/HUC10/dPL_1_11_2024/exp_EPOCH50_BS100_RHO365_HS512_trainBuff365/basin_split/"
# )
#
# # opening your root zarr folder. The mode "a" means append. You can both read and write
# root = zarr.open_group(data_split_path, mode="a")
#
# attrs_df = pd.read_csv(data_folder / "attributes.csv")
# basin_ids = attrs_df.gage_ID.values
# batch_size = 1000
# start_idx = np.arange(0, len(basin_ids), batch_size)
# end_idx = np.append(start_idx[1:], len(basin_ids))
# for idx in trange(len(start_idx), desc="reading files"):
#     try:
#         basin_ids_np = pd.read_csv(
#             qr_folder / f"Qr_{start_idx[idx]}_{end_idx[idx]}",
#             dtype=np.float32,
#             header=None,
#         ).values
#     except FileNotFoundError:
#         # sometimes the files are saved by out
#         basin_ids_np = pd.read_csv(
#             qr_folder / f"out0_{start_idx[idx]}_{end_idx[idx]}",
#             dtype=np.float32,
#             header=None,
#         ).values
#     attribute_batch_df = pd.read_csv(
#         qr_folder / "attributes" / f"attributes_{start_idx[idx]}_{end_idx[idx]}.csv"
#     )
#     attribute_batch_ids = attribute_batch_df.gage_ID.values
#     for jdx, _id in enumerate(
#         tqdm(attribute_batch_ids, desc="saving predictions separately")
#     ):
#         formatted_id = str(int(_id)).zfill(10)
#         qr = basin_ids_np[jdx:jdx + 1, :]
#         # np.save(data_split_folder / f"{formatted_id}.npy", qr)
#         # Create a Zarr dataset for this specific gage
#         # change the data type as needed, and the chunk size as you'd like.
#         # read up on chunks here: https://zarr.readthedocs.io/en/stable/tutorial.html
#         root.create_dataset(formatted_id, data=qr, chunks=(1000,), dtype="float32")