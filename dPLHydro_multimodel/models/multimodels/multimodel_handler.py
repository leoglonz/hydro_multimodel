import os

import torch.nn
from models.differentiable_model import dPLHydroModel
from models.loss_functions.get_loss_function import get_loss_func


class MultimodelHandler(torch.nn.Module):
    """
    Streamlines handling and instantiation of multiple differentiable hydrology
    models in parallel.
    """
    def __init__(self, config):
        super(MultimodelHandler, self).__init__()
        self.config = config
        self._init_models()
        
    def _init_models(self):
        """
        Initialize and store each differentiable hydro model and optimizer in
        the multimodel.
        """
        self.model_dict = dict()
        # if self.config['mode'] == 'train_wts_only':
        #     # Reinitialize trained model(s).
        #     for mod in self.config['hydro_models']:
        #         load_path = self.config[mod]
        #         self.model_dict[mod] = torch.load(load_path).to(self.config['device'])
        if self.config['use_checkpoint']:
            # Reinitialize trained model(s).
            self.all_model_params = []
            for mod in self.config['hydro_models']:
                load_path = self.config['checkpoint'][mod]
                self.model_dict[mod] = torch.load(load_path).to(self.config['device'])
                self.all_model_params += list(self.model_dict[mod].parameters())

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
            self.init_optimizer()
        elif self.config['mode'] == 'test':
            for mod in self.config['hydro_models']:
                # TODO: Get load model save path code in here!
                # load_path = os.path.join(self.config['output_dir'], self.config['forcings'], 'saved_models', '/debugging_may5_final/frozen_pnn/LSTM_E5_R365_B100_H256_n16_0/static_para/', mod + '_model_Ep' + str(self.config['epochs']) + '.pt')
                load_path = '/data/lgl5139/hydro_multimodel/dPLHydro_multimodel/runs/gages2_50/saved_models/debugging_may5_final/frozen_pnn/LSTM_E5_R365_B100_H256_n16_0/static_para/marrmot_PRMS_model_Ep5.pt'
                self.model_dict[mod] = torch.load(load_path).to(self.config['device']) 
        else:
            # Initializing differentiable hydrology model(s) and bulk optimizer.
            self.all_model_params = []
            for mod in self.config['hydro_models']:
                self.model_dict[mod] = dPLHydroModel(self.config, mod).to(self.config['device'])
                self.all_model_params += list(self.model_dict[mod].parameters())

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
            self.init_optimizer()

    def init_loss_func(self, obs) -> None:
        self.loss_func = get_loss_func(self.config, obs)
        self.loss_func = self.loss_func.to(self.config['device'])

    def init_optimizer(self) -> None:
        self.optim = torch.optim.Adadelta(self.all_model_params)

    def forward(self, dataset_dict_sample, eval=False):        
        # Batch running of the differentiable models in parallel
        self.flow_out_dict = dict()
        self.dataset_dict_sample = dataset_dict_sample

        for mod in self.model_dict:
            if eval: self.model_dict[mod].eval()  # For testing.

            # Forward each diff hydro model.
            self.flow_out_dict[mod] = self.model_dict[mod](dataset_dict_sample)

        # print(self.flow_out_dict['HBV']['BFI_sim'])
        # exit()

        return self.flow_out_dict

    def calc_loss(self, loss_dict) -> None:
        total_loss = 0
        for mod in self.model_dict:
            loss = self.loss_func(self.config,
                                  self.flow_out_dict[mod],
                                  self.dataset_dict_sample['obs'],
                                  igrid=self.dataset_dict_sample['iGrid']
                                  )
            # self.model_dict[mod].zero_grad()

            total_loss += loss
            loss_dict[mod] += loss.item()
        
        # total_loss.backward()
        # self.optim.step()

        return total_loss, loss_dict
    
