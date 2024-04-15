import torch.nn

from models.hydro_models.marrmot_PRMS.prms_marrmot import prms_marrmot
from models.hydro_models.marrmot_PRMS_gw0.prms_marrmot_gw0 import prms_marrmot_gw0
from models.hydro_models.HBV.HBVmul import HBVMul
from models.hydro_models.SACSMA.SACSMAmul import SACSMAMul
from models.hydro_models.SACSMA_with_snowpack.SACSMA_snow_mul import SACSMA_snow_Mul
from models.differentiable_model import dPLHydroModel
from models.loss_functions.get_loss_function import get_loss_func


from models.neural_networks.LSTM_models import CudnnLstmModel
from models.neural_networks.MLP_models import MLPmul



class MultimodelHandler(torch.nn.Module):
    """
    Streamlines handling and instantiation of multiple differentiable hydrology
    models in parallel.
    """
    def __init__(self, config):
        super(MultimodelHandler, self).__init__()
        self.config = config
        self.init_models()
        
    def init_models(self):
        """
        Initialize and store each differentiable hydro model and optimizer in
        the multimodel.
        """
        self.model_dict = dict()
        self.optim_dict = dict()

        # Initializing differentiable hydrology model(s) and their optimizer(s).
        for mod in self.config['hydro_models']:
            self.model_dict[mod] = dPLHydroModel(self.config, mod).to(self.config['device'])
            self.optim_dict[mod] = torch.optim.Adadelta(self.model_dict[mod].parameters())

    def init_loss_func(self, obs):
        self.loss_func = get_loss_func(self.config, obs)
        self.loss_func = self.loss_func.to(self.config['device'])
    
    def calc_ep_loss(self, ep_loss_dict):
        self.total_loss = 0

        for mod in self.model_dict:
            loss = self.loss_func(self.config,
                                                 self.flow_out_dict[mod],
                                                 self.dataset_dict_sample['obs'],
                                                 igrid=self.dataset_dict_sample['iGrid']
                                                  )
            loss.backward()
            self.optim_dict[mod].step()
            self.model_dict[mod].zero_grad()
            ep_loss_dict[mod] += loss.item()

            self.total_loss += loss.item()
        return ep_loss_dict

    def forward(self, dataset_dict_sample):        
        # Batch running of the differentiable models in parallel
        self.flow_out_dict = dict()
        self.dataset_dict_sample = dataset_dict_sample

        for mod in self.model_dict:
            # Forward each diff hydro model.
            self.flow_out_dict[mod] = self.model_dict[mod](dataset_dict_sample)

        return self.flow_out_dict
