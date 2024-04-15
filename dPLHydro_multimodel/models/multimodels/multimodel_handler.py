import torch.nn

from models.hydro_models.marrmot_PRMS.prms_marrmot import prms_marrmot
from models.hydro_models.marrmot_PRMS_gw0.prms_marrmot_gw0 import prms_marrmot_gw0
from models.hydro_models.HBV.HBVmul import HBVMul
from models.hydro_models.SACSMA.SACSMAmul import SACSMAMul
from models.hydro_models.SACSMA_with_snowpack.SACSMA_snow_mul import SACSMA_snow_Mul

from models.neural_networks.LSTM_models import CudnnLstmModel
from models.neural_networks.MLP_models import MLPmul



class MultimodelHandler(torch.nn.Module):
    """
    Streamlines handling and instantiation of multiple differentiable hydrology
    models in parallel.
    """
    def __init__(self, config, model_name):
        super(MultimodelHandler, self).__init__()
        self.config = config


