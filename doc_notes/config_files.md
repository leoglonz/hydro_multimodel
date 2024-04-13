
- Using 1 config file per model in the ensemble.



----

##### This is an example .yaml file for how you should make your config file
##### See below for what you need to have
##### Make sure the paths are correct. They are individual to your machine
randomseed: [0]   # None means random
Action: [0,1]   # 0:Train,   1: Test
device: "cuda"
nmul: 16 #8  # For parallelization.

## Directories of forcings, attributes, and output.
# For debugging, only 50 sites.
# forcing_path: 'F:\code_repos\water\data\PGML_STemp_Data\50_sites\forcing_50_20231113.feather'
# attr_path: 'F:\code_repos\water\data\PGML_STemp_Data\50_sites\attr50.feather'
# output_model: 'F:\code_repos\water\data\model_runs\PGML_STemp_results\50_sites'

## For full-scale test with temperature, 415 basins.
# forcing_path: 'D:\\data\\PGML_STemp_Data\\415_sites\\forcing_415_20231113.feather'
# attr_path: 'D:\\data\\PGML_STemp_Data\\415_sites\\attr_415_tmean_ccov.feather'
# output_model: 'D:\\data\\model_runs\\PGML_STemp_results\\models\\PRMS_SNTEMP\\415_sites'

## For full-scale test w/o tmp, 671 basins. Use "dp" attributes to
## emulate Dapeng paper resuls.
forcing_path: 'F:\code_repos\water\data\Camels\camels_671_2023113_prep\camels_671_dp_20231113.feather'
attr_path: 'F:\code_repos\water\data\Camels\camels_671_2023113_prep\attr_camels_all_sep14_2023.feather'
output_model: 'F:\code_repos\water\data\model_runs\PGML_STemp_results\671_sites\dynamic' 
# forcing_path: '../../../../../content/drive/MyDrive/Colab/data/Camels/camels_671_2023113_prep/camels_671_dp_20231113.feather'
# attr_path: '../../../../../content/drive/MyDrive/Colab/data/Camels/camels_671_2023113_prep/attr_camels_all_sep14_2023.feather'
# output_model: '../../../../../content/drive/MyDrive/Colab/data/model_runs/PGML_STemp_results/models/PRMS_SNTEMP/671_sites_dp'

## If you have a checkpoint file for a previous run, set the path to directory here. 
# saved_epoch corresponds to the .pt model save file, and is the last epoch 
# to have been fully run and saved (i.e., you now want to run saved_epoch+1). 
# saved_epoch: 40
# checkpoint_path: 'F:\code_repos\water\data\model_runs\PGML_STemp_results\671_sites\dynamic\LSTM_HBV_E50_R365_B100_H256_tr1980_1995_n16_0\parBETA_parBETAET_'
# checkpoint_path: '/content/drive/MyDrive/Colab/model_runs/PGML_STemp_results/models/PRMS_SNTEMP/671_sites/LSTM_marrmot_PRMS_E100_R365_B100_H256_tr1980_1995_n16_0'


## neural network configuration.
NN_model_name: "LSTM"   # it can be "MLP", "LSTM" too.
loss_function: "RmseLoss_flow_comb"      # "RmseLoss_flow_temp", "RmseLoss_flow_temp_BFI",  "RmseLoss_flow_temp_BFI_PET", "RmseLoss_flow_comb"  RmseLoss_BFI_temp  NSEsqrtLoss_flow_temp
loss_function_weights:
    w1: 11.0      # w1: flow loss weight (11.0), for RmseLoss_BFI_temp(w1=0.05, w2=1.0)
    w2: 1.0
# USGS code.
target: ["00060_Mean"]    #, "00010_Mean"]#, "BFI_AVE"] #, "BFI_AVE", "PET"]     
tRange: [19800101, 20101001]   # This is the entire training/testing period (i.e., beginning of training to end of testing.)
# There should be an overlap because of the warm_up.
t_train: [19801001, 19951001]   #[20080101, 20230101]   
t_val: [20101001, 20111001]   #### Currently not used.
t_test: [19951001, 20101001]    #[19990101, 20080101]  # the first year is for warm_up
warm_up: 365
rho: 365
batch_size: 100
EPOCHS: 50
hidden_size: 256
dropout: 0.5
saveEpoch: 2
no_basins: 25   # number of basins sampled in testing
varT_NN: ['prcp(mm/day)', 'tmean(C)', 'PET_hargreaves(mm/day)']  #['prcp(mm/day)', 'tmean(C)', 'dayl(s)', 'PET_hargreaves(mm/day)']   # 'dayl(s)', , 'tmin(C)', 'vp(Pa)','srad(W/m2)', pet_nldas
varC_NN: [
#  'aridity', 'p_mean', 'ETPOT_Hargr', 'NDVI', 'FW', 'SLOPE_PCT', 'SoilGrids1km_sand',
#  'SoilGrids1km_clay', 'SoilGrids1km_silt', 'glaciers', 'HWSD_clay', 'HWSD_gravel',
#  'HWSD_sand', 'HWSD_silt', 'ELEV_MEAN_M_BASIN', 'meanTa', 'permafrost',
#  'permeability','seasonality_P', 'seasonality_PET', 'snow_fraction',
#  'snowfall_fraction','T_clay','T_gravel','T_sand', 'T_silt','Porosity',
#  "DRAIN_SQKM",
  ##############
  # 'ELEV_MEAN_M_BASIN',  'ELEV_STD_M_BASIN',
  # 'SLOPE_PCT', 'DRAIN_SQKM', 'NDAMS_2009', 'MAJ_NDAMS_2009',
  # 'FRAGUN_BASIN', 'FORESTNLCD06',  'AWCAVE', 'PERMAVE', 'BDAVE',
  # 'ROCKDEPAVE', 'CLAYAVE', 'SILTAVE', 'SANDAVE', 'HGA', 'HGB',
  # 'HGC', 'HGVAR', 'HGD', 'PPTAVG_BASIN', 'SNOW_PCT_PRECIP',
  # 'PRECIP_SEAS_IND', 'T_AVG_BASIN',  'T_MAX_BASIN',
  # 'T_MAXSTD_BASIN', 'RH_BASIN',
  # 'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_DOM',
  #  'HIRES_LENTIC_PCT', 'PERDUN', 'PERHOR', 'RIP100_FOREST'
  ##################################
 'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity',
 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
 'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac',
 'dom_land_cover', 'root_depth_50', 'soil_depth_pelletier',
 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac',
 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy',
 'geol_permeability'
]

### process based stream temperature model configuration
temp_model_name: "None"       #   "None" means no temperature model, "SNTEMP", "SNTEMP_gw0"
routing_temp_model: True
res_time_type: "SNTEMP"   #"van Vliet"  # "van Vliet", "Meisner", "SNTEMP"
res_time_lenF_ssflow: 30   #60
res_time_lenF_bas_shallow: 180 #
res_time_lenF_gwflow: 730 #
res_time_lenF_srflow: 1
lat_temp_adj: True
shade_smoothening: False
frac_smoothening_mode: False
frac_smoothening_gw_filter_size: 40
frac_smoothening_ss_filter_size: 40
STemp_default_emissivity_veg: 0
STemp_default_albedo: 0.1
STemp_default_delta_Z: 1.0       # for streambed conduction
params_water_density: 1000
params_C_w: 4184    # J/kg. K
NEARZERO: 1e-5
Epan_coef: 1.0
initial_values_shade_fraction: 0.1
varT_temp_model: ['tmax(C)', 'tmin(C)',  "ccov", "PET_hargreaves(mm/day)",
              'prcp(mm/day)', 'vp(Pa)', "srad(W/m2)", 'tmean(C)']    #'dayl(s)',  , "dayofyear", "t_monthly(C)",
varC_temp_model: ["DRAIN_SQKM", "stream_length_square", 'SLOPE_PCT',
              'ELEV_MEAN_M_BASIN']       #  "area_gages2",   "RIP100_FOREST", "lat"   'PPTAVG_BASIN',
dyn_params_list_temp: ["width_coef_pow", "w1_shade"]

### process based streamflow model configurations
hydro_model_name: "HBV"  # it can be marrmot_PRMS, HBV, SACSMA, SACSMA_with_snow, marrmot_PRMS_gw0 or None -> "None" means no hydrology model
varT_hydro_model: ['prcp(mm/day)', 'tmean(C)', 'tmax(C)', 'tmin(C)', "PET_hargreaves(mm/day)"] #, "t_monthly(C)", 'dayl(s)', "dayofyear"]    # , "pet_nldas"
varC_hydro_model: ['DRAIN_SQKM', "lat"]    # "area_gages2"
routing_hydro_model: True
# if hydro_model_name = None --> we need a path to read the flow data from
# flow_data_path: "G:\\Farshid\\PGML_STemp_results\\inputs\\MSWET_MSWX_2003_GAGES_II_1384_merit\\gages-II\\npy\\hydro_sim_50_only_flow_0823.npy"
dyn_params_list_hydro: ['parBETA', 'parBETAET']   #["scx", "cgw", "alpha", "tt"]   # parTT
### PET model configurations
potet_module: "dataset" # "potet_hamon"  # "potet_hargreaves" , "dataset" -> Hamon is not ready beacause of lack of monthly coef
potet_dataset_name: "PET_hargreaves(mm/day)"  #"PET_hargreaves(mm/day)" # if "potet_module" == "dataset"

#srunoff_module: "srunoff_smidx"   # the nonlinear method, "srunoff_carea" linear method
#Dprst_flag: 1      # Dprst_flag = 1 means depression storage simulation is computed ( 0= no, 1 = yes)
#static_params_list_SNTEMP: [0,1,2,3,4,5, 6,7, 8, 9]
#semi_static_params_list_SNTEMP: []
#method_for_semi_static_param_SNTEMP: ["average", "average", "average", "average"]
#interval_for_semi_static_param_SNTEMP: [5, 5]
#static_params_list_prms: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17, 18, 19, 20]
#semi_static_params_list_prms: []
#method_for_semi_static_param_prms: ["average", "average", "average", "average"]
#interval_for_semi_static_param_prms: [30, 30]


