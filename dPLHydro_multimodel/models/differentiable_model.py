import torch.nn
from models.hydro_models.HBV.HBVmul import HBVMul
from models.hydro_models.marrmot_PRMS.prms_marrmot import prms_marrmot
from models.hydro_models.marrmot_PRMS_gw0.prms_marrmot_gw0 import \
    prms_marrmot_gw0
from models.hydro_models.SACSMA.SACSMAmul import SACSMAMul
from models.hydro_models.SACSMA_with_snowpack.SACSMA_snow_mul import \
    SACSMA_snow_Mul
from models.neural_networks.lstm_models import CudnnLstmModel
from models.neural_networks.mlp_models import MLPmul


class dPLHydroModel(torch.nn.Module):
    """
    Default class for instantiating a differentiable hydrology model 
    e.g., HBV, PRMS, SAC-SMA, and their variants.
    """
    def __init__(self, config, model_name):
        super(dPLHydroModel, self).__init__()
        self.config = config
        self.model_name = model_name 
        self._init_model()

    def _init_model(self):
        # hydro_model_initialization
        if self.model_name == 'marrmot_PRMS':
            self.hydro_model = prms_marrmot()
        elif self.model_name == 'marrmot_PRMS_gw0':
            self.hydro_model = prms_marrmot_gw0()
        elif self.model_name == 'HBV':
            self.hydro_model = HBVMul(self.config)
        elif self.model_name == 'SACSMA':
            self.hydro_model = SACSMAMul()
        elif self.model_name == 'SACSMA_with_snow':
            self.hydro_model = SACSMA_snow_Mul()
        else:
            raise ValueError(self.model_name, "is not a valid hydrology model.")
    
        # Get dim of NN model based on hydro model
        self.get_nn_model_dim()
        # NN_model_initialization
        if self.config['nn_model'] == 'LSTM':
            self.NN_model = CudnnLstmModel(nx=self.nx,
                                           ny=self.ny,
                                           hiddenSize=self.config['hidden_size'],
                                           dr=self.config['dropout'])
        elif self.config['nn_model'] == 'MLP':
            self.NN_model = MLPmul(self.args, nx=self.nx, ny=self.ny)
        else:
            raise ValueError(self.config['nn_model'], "is not a valid neural network type.")

    def get_nn_model_dim(self) -> None:
        self.nx = len(self.config['observations']['var_t_nn'] + self.config['observations']['var_c_nn'])

        # output size of NN
        if self.config['routing_hydro_model'] == True:
            self.ny = self.config['nmul'] * (len(self.hydro_model.parameters_bound)) + len(
                self.hydro_model.conv_routing_hydro_model_bound)
        else:
            self.ny = self.args['nmul'] * len(self.hydro_model.parameters_bound)

    def breakdown_params(self, params_all) -> None:
        params_dict = dict()
        params_hydro_model = params_all[:, :, :self.ny]

        # hydro params
        params_dict['hydro_params_raw'] = torch.sigmoid(
            params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.config['nmul']]).view(
            params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
            self.config['nmul'])
        # routing params
        if self.config['routing_hydro_model'] == True:
            params_dict['conv_params_hydro'] = torch.sigmoid(
                params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.config['nmul']:])
        else:
            params_dict['conv_params_hydro'] = None
        return params_dict

    def forward(self, dataset_dict_sample) -> None:
        params_all = self.NN_model(dataset_dict_sample['inputs_nn_scaled'])

        # Breaking down params into different pieces for different models (PET, hydro)
        params_dict = self.breakdown_params(params_all)
        
        # Hydro model
        flow_out = self.hydro_model(
            dataset_dict_sample['x_hydro_model'],
            dataset_dict_sample['c_hydro_model'],
            params_dict['hydro_params_raw'],
            self.config,
            warm_up=self.config['warm_up'],
            routing=self.config['routing_hydro_model'],
            conv_params_hydro=params_dict['conv_params_hydro']
        )
        # Baseflow index percentage
        ## Using two deep groundwater buckets: gwflow & bas_shallow
        if 'bas_shallow' in flow_out.keys():
            baseflow = flow_out['gwflow'] + flow_out['bas_shallow']
        else:
            baseflow = flow_out['gwflow']
        flow_out['BFI_sim'] = 100 * (torch.sum(baseflow, dim=0) / (
                torch.sum(flow_out['flow_sim'], dim=0) + 0.00001))[:, 0]
        
        return flow_out
    