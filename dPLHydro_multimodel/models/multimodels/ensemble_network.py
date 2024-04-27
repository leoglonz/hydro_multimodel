import models.multimodels.mm_functional as F
import torch
from models.loss_functions.get_loss_function import get_loss_func
from models.neural_networks.lstm_models import CudnnLstmModel
from utils import master as m

# Set global torch device and dtype.
device, dtype = m.set_globals()



class EnsembleWeights(torch.nn.Module):
    """
    Interface for weighting neural network used for ensembling multiple
    hydrology models.
    """
    def __init__(self, config):
        super(EnsembleWeights, self).__init__()
        self.config = config
        self.name = 'Ensemble Weighting Network'
        self.get_model()
    
    def get_model(self) -> None:
        """
        Initialize LSTM and optimizer.
        """
        self.get_nn_model_dim()

        self.lstm = CudnnLstmModel(nx=self.nx,
                                      ny=self.ny,
                                      hiddenSize=self.config['weighting_nn']['hidden_size'],
                                      dr=self.config['weighting_nn']['dropout']
                                      ).to(F.device)
        self.optim = torch.optim.Adadelta(self.lstm.parameters())
        self.lstm.zero_grad()
        self.lstm.train()
    
    def init_loss_func(self, obs):
        self.loss_func = get_loss_func(self.config['weighting_nn'], obs)
        self.loss_func = self.loss_func.to(F.device)

    def get_nn_model_dim(self) -> None:
        self.nx = len(self.config['observations']['var_t_nn'] + self.config['observations']['var_c_nn'])
        self.ny = len(self.config['hydro_models'])#output size of NN

    def forward(self, dataset_dict_sample):
        self.dataset_dict_sample = dataset_dict_sample

        # Get scaled mini-batch of basin forcings + attributes.
        nn_inputs = dataset_dict_sample['inputs_nn_scaled'].requires_grad_(True)

        # Forward lstm to get model weights.
        self.weights = self.lstm(nn_inputs)

    def get_loss(self, hydro_preds, ep_loss):
        # total loss is the sum of range bound loss, and streamflow ensemble preds vs obs.
        ntstep = self.weights.shape[0]
        ngage = self.weights.shape[1]

        # Scale weights.
        if self.config['weighting_nn']['method'] == 'sigmoid':
            self.weights_scaled = torch.sigmoid(self.weights)
        elif self.config['weighting_nn']['method'] == 'softmax':
            self.weights_scaled = torch.softmax(self.weights)
        else:
            raise ValueError(self.config['weighting_nn']['method'], "is not a valid model weighting method.")

        # Loss on weights.
        self.calc_range_bound_loss()

        # Get ensembled streamflow.
        self.ensemble_pred = torch.zeros((ntstep, ngage), requires_grad=True, dtype=torch.float32).to(F.device)
        for mod in range(self.weights.shape[2]):
            self.ensemble_pred += self.weights_scaled[:, :, mod] * hydro_preds[:, :, mod]
        # torch.sum(hydro_preds * weights_scaled, dim=2)

        # Loss on streamflow preds.
        loss_sf = self.loss_func(self.ensemble_pred, self.dataset_dict_sample['obs'])

        total_loss = self.range_bound_loss + loss_sf

        total_loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        ep_loss += total_loss.item()

        return ep_loss
