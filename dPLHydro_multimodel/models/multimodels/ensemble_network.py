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
        self._init_model()
        self.range_bound_loss = F.RangeBoundLoss(config)
    
    def _init_model(self):
        """
        Initialize LSTM.
        """
        self.get_nn_model_dim()

        self.lstm = CudnnLstmModel(nx=self.nx,
                                      ny=self.ny,
                                      hiddenSize=self.config['weighting_nn']['hidden_size'],
                                      dr=self.config['weighting_nn']['dropout']
                                      ).to(F.device)
        # self.optim = torch.optim.Adadelta(self.lstm.parameters()) 
        # Save model parameters to pass to optimizer
        self.model_params = self.lstm.parameters()
        self.lstm.zero_grad()
        self.lstm.train()
    
    def init_loss_func(self, obs) -> None:
        self.loss_func = get_loss_func(self.config['weighting_nn'], obs)
        self.loss_func = self.loss_func.to(self.config['device'])

    def init_optimizer(self):
        self.optim = torch.optim.Adadelta(self.lstm.parameters())

    def get_nn_model_dim(self) -> None:
        self.nx = len(self.config['observations']['var_t_nn'] + self.config['observations']['var_c_nn'])
        self.ny = len(self.config['hydro_models'])  # Output size of NN

    def forward(self, dataset_dict_sample, eval=False) -> None:
        self.dataset_dict_sample = dataset_dict_sample

        # Get scaled mini-batch of basin forcings + attributes.
        # inputs_nn_scaled = x_nn + c_nn, forcings + basin attributes
        nn_inputs = dataset_dict_sample['inputs_nn_scaled'].requires_grad_(True)

        if eval: self.lstm.eval()  # For testing
        self.weights = self.lstm(nn_inputs) # Forward

        self.weights_dict = dict()
        for i, mod in enumerate(self.config['hydro_models']):
            # Extract predictions into model dict + remove warmup period from output.
            self.weights_dict[mod] = self.weights[self.config['warm_up']:,:,i]
            

    def calc_loss(self, hydro_preds, loss_dict=None) -> None:
        """
        Computes composite loss: 
        1) Takes in predictions from set of hydro models, and computes a loss on the linear combination of model predictions using lstm-derived weights.

        2) Calculates range-bound loss on the lstm weights.
        """
                
        ntstep = self.weights.shape[0] ### TODO replace weights with weights_dict.
        ngage = self.weights.shape[1]

        # Scale weights.
        if self.config['weighting_nn']['method'] == 'sigmoid':
            self.weights_scaled = torch.sigmoid(self.weights)
        elif self.config['weighting_nn']['method'] == 'softmax':
            self.weights_scaled = torch.softmax(self.weights)
        else:
            raise ValueError(self.config['weighting_nn']['method'], "is not a valid model weighting method.")

        # Range-bound loss on weights.
        weights_sum = torch.sum(self.weights_scaled, dim=2)
        loss_rb = self.range_bound_loss([weights_sum])

        # Get ensembled streamflow.
        self.ensemble_pred = torch.zeros((ntstep, ngage), dtype=torch.float32, device=self.config['device'])

        for i, mod in enumerate(self.config['hydro_models']):  
            h_pred = hydro_preds[mod]['flow_sim'][:, :].squeeze()
            if self.weights_scaled.size(0) != h_pred.size(0):
                # Cut out warmup data present when testing model from loaded mod file.
                h_pred = h_pred[self.config['warm_up']:,:]

            self.ensemble_pred += self.weights_scaled[:, :, i] * h_pred 
        # torch.sum(hydro_preds * weights_scaled, dim=2)

        # Loss on streamflow preds.
        loss_sf = self.loss_func(self.config,
                                 self.ensemble_pred,
                                 self.dataset_dict_sample['obs'],
                                 igrid=self.dataset_dict_sample['iGrid']
                                 )
        # self.lstm.zero_grad()

        # Return total_loss for optimizer.
        total_loss = loss_rb + loss_sf
        if loss_dict:
            loss_dict['wtNN'] += total_loss.item()
            return total_loss, loss_dict

        # total_loss.backward()
        # self.optim2.step()
        # self.optim2.zero_grad()
        # comb_loss += total_loss.item()
        # return comb_loss

        return total_loss
    
    