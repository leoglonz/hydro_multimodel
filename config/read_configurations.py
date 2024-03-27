"""
Code to read the config files for each hydro model variation.
We keep them separated so that each model can easily be setup as desired.  
"""

import os
try: 
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    print("YAML Module not found.")



### Put your paths to the config files on your local machine here.
config_path_hbv = "config/config_HBV.yaml"
config_path_prms = "config/config_PRMS.yaml"
config_path_sacsma= "config/config_SACSMA.yaml"
config_path_sacsma_snow= "config/config_SACSMA_with_snow.yaml"
config_path_hbv_hy= "config/config_hbv_hydrodl.yaml"


yaml = YAML(typ="safe")
path_hbv = os.path.realpath(config_path_hbv)
path_prms = os.path.realpath(config_path_prms)
path_sacsma = os.path.realpath(config_path_sacsma)
path_sacsma_snow = os.path.realpath(config_path_sacsma_snow)
path_hbv_hy = os.path.realpath(config_path_hbv_hy)


stream_hbv = open(path_hbv, "r")
stream_prms = open(path_prms, "r")
stream_sacsma = open(path_sacsma, "r")
stream_sacsma_snow = open(path_sacsma_snow, "r")
stream_hbv_hy = open(path_hbv_hy, "r")


config_hbv = yaml.load(stream_hbv)
config_prms = yaml.load(stream_prms)
config_sacsma = yaml.load(stream_sacsma)
config_sacsma_snow = yaml.load(stream_sacsma_snow)
config_dplhbv_dyn = yaml.load(stream_hbv_hy)
config_dplhbv_stat = None  # Not implemented yet.

master_config = {'HBV':config_hbv,
                 'dPLHBV_stat': config_dplhbv_stat,
                 'dPLHBV_dyn': config_dplhbv_dyn, 
                 'SACSMA': config_sacsma,
                 'SACSMA_snow': config_sacsma_snow,
                 'marrmot_PRMS':config_prms}
