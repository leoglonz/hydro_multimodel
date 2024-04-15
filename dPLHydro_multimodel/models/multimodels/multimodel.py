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
