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


import torch
import torch.nn as nn
import numpy as np
import multim_functional as F
from hydroDL_depr.model import rnn

from hydroDL_depr.model.rnn import CudnnLstm, CudnnLstmModel

# Global variables:
device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32



def createTensor(dims, requires_grad=False):
    """
    A small function to centrally manage device, data types, etc., of new arrays
    """
    return torch.zeros(dims,requires_grad=requires_grad,dtype=dtype).to(device)


def createDictFromKeys(keyList, mtd=0, dims=None, dat=None):
    """
    A modular dictionary initializer from C. Shen.

    mtd = 
        0: Init keys to None,
        1: Init keys to zero tensors,
        11: Init keys to tensors with the same vals as `dat`,
        2: Init keys to slices of `dat`,
        21: Init keys with cloned slices of `dat`.
    """
    d = {}
    for kk, k in enumerate(keyList):
        if mtd == 0 or mtd is None or mtd == "None":
            d[k] = None
        elif mtd == 1 or mtd == 'zeros':
            d[k] = createTensor(dims)
        elif mtd == 11 or mtd == 'value':
            d[k] = createTensor(dims) + dat
        elif mtd == 2 or mtd == 'ref':
            d[k] = dat[..., kk]
        elif mtd == 21 or mtd == 'refClone':
            d[k] = dat[..., kk].clone()
    return d



class HydroMultimodel(nn.Module):
    """
    A wrapper for managing a collection of (trained) hydromodels, 
    and applyinf different ensembling methods.
    """
    def __init__(self, modelList, argList):
        super(HydroMultimodel, self).__init__()
        """
        Instantiate hydro models to be ensembled.
        """

        self.ensemble_mtd = None  # Ensemble method 
        # self.defaultKeys = {}
        # self.defaultKeys['models'] = ['HBV', 'dPLHBV', 'dPLHBV_dp', 'SACSMA', 'SACSMA_snow', 'marrmot_PRMS']
        # self.defaultKeys['ensemble_mtds'] = {}  
          
        self.modelDict = createDictFromKeys(modelList)
        self.argDict = createDictFromKeys(modelList, mtd='ref', dat=argList)
        self.initModel()


    def initModel(self, mode='all', *args, **kwargs):
        """
        Instantiate hydro models to be ensembled.
        """
        if mode == 'all':
            for mod in self.modelsDict:
                if mod in ['HBV', 'SACSMA', 'SACSMA_snow', 'marrmot_PRMS']:
                    self.modelDict[mod] = PGMLHydroModel(self.argDict[mod])
                elif mod in ['dPLHBV']:
                    self.modelDict[mod] = rnn.MultiInv_HBVModel(*args, **kwargs)
                elif mod in ['dPLHBV_dp']:
                    self.modelDict[mod] = rnn.MultiInv_HBVTDModel(*args, **kwargs)
                else:
                    raise ValueError("Invalid hydrology model specified.")     
                 
        elif mode in self.modelDict:
            if mod in ['HBV', 'SACSMA', 'SACSMA_snow', 'marrmot_PRMS']:
                self.modelDict[mod] = PGMLHydroModel(self.argDict[mod])
            elif mod in ['dPLHBV_dp']:
                self.modelDict[mod] = rnn.MultiInv_HBVModel(*args, **kwargs)
            else:
                self.modelDict[mod] = rnn.MultiInv_HBVTDModel(*args, **kwargs) 
        else:
            raise ValueError("Invalid hydrology model specified.")
        

    def multimodel_ensemble():
        mm_ensemble = MultiModelEnsemble()


    def fuse_on_avg():
        # fuse with average of streamflows here.
        x = 1


    def fuse_on_md():
        # Fuse with median of streamflows here.
        x = 1


    def forward(self, *args, **kwargs):
        for key in self.modelsDict:
            self.modelDict[key](*args, **kwargs)
      

import multim_functional as F


class WeightedEnsemble(torch.nn.Module):
    """
    LSTM model to get weights for linear combinations of multiple hydro models.
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





'''
class EnsembleWeights(torch.nn.Module):
    def __init__(self, *, ninv, hiddeninv, drinv=0.5, prcp_datatypes=1):
        super(EnsembleWeights, self).__init__()
        self.ninv = ninv
        self.prcp_datatypes = prcp_datatypes

        self.ntp = prcp_datatypes*3
        self.hiddeninv = hiddeninv

        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=self.ntp, hiddenSize=hiddeninv, dr=drinv).cuda()

        # Adjust the range for acceptable sum of weights for loss.
        # Potentially worth testing different combinations.
        lb_prcp = [0.95]
        ub_prcp = [1.05]
        self.RangeBoundLoss = RangeBoundLoss(lb=lb_prcp, ub=ub_prcp)

    def forward(self, x, prcp_loss_factor):
        x.requires_grad = True

        weights = self.lstminv(x)
        weights_scaled = torch.sigmoid(weights)

        # initialize empty
        ntstep = weights.shape[0]
        ngage = weights.shape[1]
        prcp_wavg = torch.zeros((ntstep, ngage), requires_grad=True, dtype=torch.float32).cuda()

        # get weighted avg
        for para in range(weights.shape[2]):
            prcp_wavg = prcp_wavg + weights_scaled[:, :, para] * x[:, :, para]

        # calculate loss
        prcp_weight_sum = torch.sum(weights_scaled[:,:,:self.ntp], dim=2)
        range_bound_loss_prcp = self.RangeBoundLoss([prcp_weight_sum], factor=prcp_loss_factor)

        # Use if the Dr. Shen requests gradient analysis.
        # grad_daymet = autograd.grad(outputs=wghts_scaled[:, :, 0], inputs=z, grad_outputs=torch.ones_like(wghts_scaled[:, :, 0]), retain_graph=True)[0]
        # grad_maurer = autograd.grad(outputs=wghts_scaled[:, :, 1], inputs=z, grad_outputs=torch.ones_like(wghts_scaled[:, :, 1]), retain_graph=True)[0]
        # grad_nldas = autograd.grad(outputs=wghts_scaled[:, :, 2], inputs=z, grad_outputs=torch.ones_like(wghts_scaled[:, :, 2]), retain_graph=True)[0]

        # return x_new, range_bound_loss_prcp, wghts_scaled, grad_daymet, grad_maurer, grad_nldas
        return prcp_wavg, weights_scaled, range_bound_loss_prcp
'''



















class PGMLHydroModel(torch.nn.Module):
    """
    Differentiable hydro model code from F. Rahmani PGML_STemp_with_Snow.
    Use for PRMS, SAC-SMA, and unmodified HBV.
    """
    def __init__(self, args):
        super(diff_hydro_temp_model, self).__init__()
        self.args = args
        self.get_model()

    def get_NN_model_dim(self) -> None:
        self.nx = len(self.args["varT_NN"] + self.args["varC_NN"])

        # output size of NN
        if self.args["hydro_model_name"] != "None":
            if self.args["routing_hydro_model"] == True:  # needs a and b for routing with conv method
                self.ny_hydro = self.args["nmul"] * (len(self.hydro_model.parameters_bound)) + len(
                    self.hydro_model.conv_routing_hydro_model_bound)
            else:
                self.ny_hydro = self.args["nmul"] * len(self.hydro_model.parameters_bound)
        else:
            self.ny_hydro = 0

        # SNTEMP  # needs a and b for calculating different source flow temperatures with conv method
        if self.args["temp_model_name"] != "None":
            if self.args["routing_temp_model"] == True:
                self.ny_temp = self.args["nmul"] * (len(self.temp_model.parameters_bound)) + len(
                    self.temp_model.conv_temp_model_bound)
            else:
                self.ny_temp = self.args["nmul"] * len(self.temp_model.parameters_bound)
            if self.args["lat_temp_adj"] == True:
                self.ny_temp = self.ny_temp + self.args["nmul"]
        else:
            self.ny_temp = 0
        # if self.args["hydro_model_name"] == "HBV":   # no need to have a PET to AET coef
        #     self.ny_PET = 0
        # elif self.args["hydro_model_name"] == "marrmot_PRMS":   # need a PET to AET coef
        #     self.ny_PET = self.args["nmul"]
        # if self.args["potet_module"] in ["potet_hargreaves", "potet_hamon", "dataset"]:
        #     self.ny_PET = self.args["nmul"]
        self.ny = self.ny_hydro + self.ny_temp # + self.ny_PET

    def get_model(self) -> None:
        # hydro_model_initialization
        if self.args["hydro_model_name"] != "None":
            if self.args["hydro_model_name"] == "marrmot_PRMS":
                self.hydro_model = prms_marrmot()
            elif self.args["hydro_model_name"] == "marrmot_PRMS_gw0":
                self.hydro_model = prms_marrmot_gw0()
            elif self.args["hydro_model_name"] == "HBV":
                self.hydro_model = HBVMul()
            elif self.args["hydro_model_name"] == "SACSMA":
                self.hydro_model = SACSMAMul()
            elif self.args["hydro_model_name"] == "SACSMA_with_snow":
                self.hydro_model = SACSMA_snow_Mul()
            elif self.args["hydro_model_name"] != "None":
                print("hydrology (streamflow) model type has not been defined")
                exit()
            # temp_model_initialization
        if self.args["temp_model_name"] != "None":
            if self.args["temp_model_name"] == "SNTEMP":
                self.temp_model = SNTEMP_flowSim()  # this model needs a hydrology model as backbone
            elif self.args["temp_model_name"] == "SNTEMP_gw0":
                self.temp_model = SNTEMP_flowSim_gw0()  # this model needs a hydrology model as backbone, and 4 outflow
            elif self.args["temp_model_name"] != "None":
                print("temp model type has not been defined")
                exit()
        # get the dimensions of NN model based on hydro modela and temp model
        self.get_NN_model_dim()
        # NN_model_initialization
        if self.args["NN_model_name"] == "LSTM":
            self.NN_model = CudnnLstmModel(nx=self.nx,
                                           ny=self.ny,
                                           hiddenSize=self.args["hidden_size"],
                                           dr=self.args["dropout"])
        elif self.args["NN_model_name"] == "MLP":
            self.NN_model = MLPmul(self.args, nx=self.nx, ny=self.ny)
        else:
            print("NN model type has not been defined")
            exit()

    def breakdown_params(self, params_all):
        params_dict = dict()
        params_hydro_model = params_all[-1, :, :self.ny_hydro]
        params_temp_model = params_all[-1, :, self.ny_hydro: (self.ny_hydro + self.ny_temp)]
        # if self.ny_PET > 0:
        #     params_dict["params_PET_model"] = torch.sigmoid(params_all[-1, :, (self.ny_hydro + self.ny_temp):])
        # else:
        #     params_dict["params_PET_model"] = None


        # Todo: I should separate PET model output from hydro_model and temp_model.
        #  For now, evap is calculated in both models individually (with same method)

        if self.args['hydro_model_name'] != "None":
            # hydro params
            params_dict["hydro_params_raw"] = torch.sigmoid(
                params_hydro_model[:, :len(self.hydro_model.parameters_bound) * self.args["nmul"]]).view(
                params_hydro_model.shape[0], len(self.hydro_model.parameters_bound),
                self.args["nmul"])
            # routing params
            if self.args["routing_hydro_model"] == True:
                params_dict["conv_params_hydro"] = torch.sigmoid(
                    params_hydro_model[:, len(self.hydro_model.parameters_bound) * self.args["nmul"]:])
            else:
                params_dict["conv_params_hydro"] = None

        if self.args['temp_model_name'] != "None":
            # hydro params
            params_dict["temp_params_raw"] = torch.sigmoid(
                params_temp_model[:, :len(self.temp_model.parameters_bound) * self.args["nmul"]]).view(
                params_temp_model.shape[0], len(self.temp_model.parameters_bound),
                self.args["nmul"])
            # convolution parameters for ss and gw temp calculation
            if self.args["routing_temp_model"] == True:
                params_dict["conv_params_temp"] = torch.sigmoid(params_temp_model[:, -len(self.temp_model.conv_temp_model_bound):])
            else:
                print("it has not been defined yet what approach should be taken in place of conv")
                exit()
        return params_dict


    def forward(self, dataset_dictionary_sample):
        params_all = self.NN_model(dataset_dictionary_sample["inputs_NN_scaled_sample"][self.args["warm_up"]:, :, :])
        # breaking down the parameters to different pieces for different models (PET, hydro, temp)
        params_dict = self.breakdown_params(params_all)
        if self.args['hydro_model_name'] != "None":
            # hydro model
            flow_out = self.hydro_model(
                dataset_dictionary_sample["x_hydro_model_sample"],
                dataset_dictionary_sample["c_hydro_model_sample"],
                params_dict['hydro_params_raw'],
                self.args,
                # PET_param=params_dict["params_PET_model"],  # PET is in both temp and flow model
                warm_up=self.args["warm_up"],
                routing=self.args["routing_hydro_model"],
                conv_params_hydro=params_dict["conv_params_hydro"]
            )
            # baseflow index percentage
            flow_out["BFI_sim"] = 100 * (torch.sum(flow_out["gwflow"], dim=0) / (
                    torch.sum(flow_out["flow_sim"], dim=0) + 0.00001))[:, 0]

            if self.args['temp_model_name'] != "None":
                # source flow calculation and converting mm/day to m3/ day
                source_flows_dict = source_flow_calculation(self.args, flow_out,
                                                              dataset_dictionary_sample[
                                                                  "c_NN_sample"],
                                                              after_routing=True)
                # temperature model
                temp_out = self.temp_model.forward(dataset_dictionary_sample["x_temp_model_sample"],
                                                   dataset_dictionary_sample["c_temp_model_sample"],
                                                   params_dict["temp_params_raw"],
                                                   conv_params_temp=params_dict["conv_params_temp"],
                                                   args=self.args,
                                                   PET=flow_out["PET_hydro"] * (1 / (1000 * 86400)),   # converting mm/day to m/sec,
                                                   source_flows=source_flows_dict)

                return {**flow_out, **temp_out}   # combining both dictionaries
            else:
                return flow_out



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
    

