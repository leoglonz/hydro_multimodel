# This file contains an interface for multimodel synthesis, along with 
# ensembling objects to manipulate the interface into different ensembling methods.
#
# Acknowledgements:
#  - Syntax and smaller codes reproduced from C. Shen.
#  - WeightedEnsemble code repurposed from K. Sawadekar precip fusion.
#  - HBV, SAC-SMA, Marrmot PRMS models and setup modified from F. Rahmani.
#  - dHBV model and setup from MHPI Team C. Shen, et al.
################################################################################

from config.read_configurations import config_hbv as hbvArgs
from config.read_configurations import config_hbv_hydrodl as dplhbvArgs
from config.read_configurations import config_prms as prmsArgs
from config.read_configurations import config_sacsma as sacsmaArgs
from config.read_configurations import config_sacsma_snow as sacsmaSnowArgs

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

import torch
import torch.nn as nn

from hydroDL.model_new.rnn import CudnnLstmModel
import mm_interface.mm_functional as F
from mm_interface import master as m


# Set global torch device and dtype.
device, dtype = m.set_globals()



class WeightedEnsemble(torch.nn.Module):
    """
    Interface for LSTM model to get weights for linear combinations of
    multiple hydro models.
    (Modified From K. Sawadekar)
    """
    def __init__(self, dims={'ninv':2, 'hiddeninv':128, 'drinv':0.5, 'nmodels':2}):
        # `Dims`` is a stand-in until I eventually calculate directly from input data x 
        # and c (apart from hiddeninv).
        super(WeightedEnsemble, self).__init__()
        self.name = 'WeightedEnsemble'
        self.dims = dims


        self.defaultKeys = {}
        self.defaultKeys['parameters'] = ('lowerb_loss', 'upperb_loss', 'loss_factor', 'hiddeninv', 'drinv')
        self.defaultKeys['model_states'] = ('weights', 'weights_scaled', 'weights_sum','prcp_wavg', 'range_bound_loss')
        # self.defaultKeys['attributes'] = (
        #     'ELEV_MEAN_M_BASIN',  'ELEV_STD_M_BASIN',
        #     'SLOPE_PCT', 'DRAIN_SQKM', 'NDAMS_2009', 'MAJ_NDAMS_2009',
        #     'FRAGUN_BASIN', 'FORESTNLCD06',  'AWCAVE', 'PERMAVE', 'BDAVE',
        #     'ROCKDEPAVE', 'CLAYAVE', 'SILTAVE', 'SANDAVE', 'HGA', 'HGB',
        #     'HGC', 'HGVAR', 'HGD', 'PPTAVG_BASIN', 'SNOW_PCT_PRECIP',
        #     'PRECIP_SEAS_IND', 'T_AVG_BASIN',  'T_MAX_BASIN',
        #     'T_MAXSTD_BASIN', 'RH_BASIN',
        #     'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_DOM',
        #     'HIRES_LENTIC_PCT', 'PERDUN', 'PERHOR', 'RIP100_FOREST'
        # )
        # Initialize weighting model
        self.initModel()
    

    def initModel(self):
        """
        Initialize LSTM and any other required models.
        """
        self.lstminv = CudnnLstmModel(nx=self.dims['ninv'], ny=self.dims['nmodels'], hiddenSize=self.dims['hiddeninv'], dr=self.dims['drinv'])


    def preRun(self, x, c, loss_factor):
        """
        Initialize data structures, some variables, and parameters.
        """
        #### attributes `c` not currently used yet. Need to pipe this into LSTM.
        x.requires_grad = True
        self.prcp = x

        dKeys = self.defaultKeys
        # self.prcp = createDictFromKeys(dKeys['models'], mtd='ref', dat=x)
        # self.attributes = createDictFromKeys(dKeys['attributes'], mtd='ref', dat=x)
        self.params = createDictFromKeys(dKeys['parameters'], dims=1, mtd='zeros')
        self.mstates = createDictFromKeys(dKeys['model_states'], dims=1, mtd='zeros')
        self.attributes = createDictFromKeys(dKeys['attributes'], dims=1, mtd=x.attributes)

        self.initDims()
        self.initParams()

        self.params['loss_factor'] = loss_factor


    def initDims(self):
        self.dims['ntstep'], self.dims['ngage'] = self.states['prcp_weights'].shape


    def initParams(self):
        # Adjust the range for acceptable sum of weights for loss.
        self.params['lowerb_loss'] = [0.95]
        self.params['upperb_loss'] = [1.05]
        # self.params['loss_factor'] = 15


    def range_bound_loss(self):
        """
        Calculate a loss to limit parameters from exceeding a specified range.
        """
        self.weightSum()

        self.range_bound_loss = F.range_bound_loss(self.mstates['weights_sum'], scale_factor=self.params['loss_factor'])


    def getWeights(self):
        self.mstates['weights'] = self.lstminv(self.prcp)
        self.mstates['weights_scaled'] = torch.sigmoid(self.states['weights'])

        self.weightAvg()


    def weightedAvg(self):
        self.mstates['prcp_wavg'] = F.weighted_avg(self.prcp, self.mstates['weights'], self.mstates['weights_scaled'], (self.dims['ntstep'],self.dims['ngage']))
    

    def weightSum(self):
        """
        For loss calculation.
        """
        self.mstates['weights_sum'] = F.t_sum(self.mstates['weights_scaled'], self.dims['nmodels'], self.dims['ninv'])


    def forward(self, x, c, loss_factor=15):
        self.preRun(x, c, loss_factor)   ### Need work here to get basin attributes that can be used in LSTM ()    

        self.getWeights()  
        self.range_bound_loss()  # Compute loss





if __name__ == "__main__":
    # List hydro models to combine in multimodel.
    models = ['dPLHBV_dp', 'SACSMA_snow', 'marrmot_PRMS']
    args_list = [dplhbvArgs, sacsmaSnowArgs, prmsArgs]
    # HBV: unmodified HBV,
    # dPLHBV_dp: delta HBV with dynamic parameters,
    # SACSMA: traditional SAC-SMA,
    # SACSMA_snow: SAC-SMA with snow/melting module,
    # marrmot_PRMS: ..

    # Instantiating multimodel carrier object.
    HydroMultimodel(models)


    forType = 'daymet'
    Ttrain = [19801001, 19951001] #training period
    Ttest = [19951001, 20101001]  # Testing period'

    # Define inputs
    if forType == 'daymet':
        varF = ['dayl', 'prcp', 'srad', 'tmax', 'tmin', 'tmean', 'vp']
    else:
        varF = ['dayl', 'prcp', 'srad', 'tmax', 'vp']

    # Set list of attributes (varC_NN) for GAGESII dataset. Use for full 671 CAMELS Basins.
    attrLst = [
        'ELEV_MEAN_M_BASIN',  'ELEV_STD_M_BASIN',
        'SLOPE_PCT', 'DRAIN_SQKM', 'NDAMS_2009', 'MAJ_NDAMS_2009',
        'FRAGUN_BASIN', 'FORESTNLCD06',  'AWCAVE', 'PERMAVE', 'BDAVE',
        'ROCKDEPAVE', 'CLAYAVE', 'SILTAVE', 'SANDAVE', 'HGA', 'HGB',
        'HGC', 'HGVAR', 'HGD', 'PPTAVG_BASIN', 'SNOW_PCT_PRECIP',
        'PRECIP_SEAS_IND', 'T_AVG_BASIN',  'T_MAX_BASIN',
        'T_MAXSTD_BASIN', 'RH_BASIN',
        'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_DOM',
        'HIRES_LENTIC_PCT', 'PERDUN', 'PERHOR', 'RIP100_FOREST'
    ]
    # CAMELS dataset attributes.
    # attrLst = [
    #     'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity',
    #     'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
    #     'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
    #     'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac',
    #     'dom_land_cover', 'root_depth_50', 'soil_depth_pelletier',
    #     'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    #     'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac',
    #     'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
    #     'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy',
    #     'geol_permeability'
    # ]
    

