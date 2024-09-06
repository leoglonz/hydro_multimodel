import netCDF4 as nc
folder = "/data/yxs275/DPL_HBV/Global_data/Hylke/GSCD_v2.0/"
file_path = folder+"/catchments.txt"

# Open the file in read mode
with open(file_path, "r") as file:
    # Read all lines into a list
    lines = file.readlines()

file_path = folder+ "GSCD_v2.0.nc"
with nc.Dataset(file_path, "r") as nc_file:
    variable_data = nc_file.variables['variable_name'][:]
print("Done")