import os

import torch.nn
from models.differentiable_model import dPLHydroModel
from models.loss_functions.get_loss_function import get_loss_func


class ModelHandler(torch.nn.Module):
    """
    Streamlines handling and instantiation of multiple differentiable hydrology
    models in parallel.

    Also capable of running a single hydro model.
    """
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self._init_models()
        
    def _init_models(self):
        """
        Initialize and store each differentiable hydro model and optimizer in
        the multimodel.
        """
        self.model_dict = dict()

        ### TODO: Modularize these experiment modes so that people can more easily insert their own custom experiments
        ### (unless they just decide to edit the train.py file)
        if (self.config['ensemble_type'] == 'none') and (len(self.config['hydro_models']) > 1):
            raise ValueError("Multiple hydro models given, but ensemble type is not specified. Check configurations.")
        elif self.config['mode'] == 'train_wtnn_only':
            # Reinitialize trained model(s).
            for mod in self.config['hydro_models']:
                self.model_dict[mod] = torch.load(load_path).to(self.config['device'])
        if self.config['use_checkpoint']:
            # Reinitialize trained model(s).
            self.all_model_params = []
            for mod in self.config['hydro_models']:
                load_path = self.config['checkpoint'][mod]
                self.model_dict[mod] = torch.load(load_path).to(self.config['device'])
                self.all_model_params += list(self.model_dict[mod].parameters())

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
            # Note: optimizer init must be within this handler, and not called
            # externally, so that it can be wrapped by a CSDMS BMI (NextGen comp.)
            self.init_optimizer()
        elif self.config['mode'] in ['test', 'test_bmi']:
            for mod in self.config['hydro_models']:
                self.load_model(mod)
        else:
            # Initialize differentiable hydrology model(s) and bulk optimizer.
            self.all_model_params = []
            for mod in self.config['hydro_models']:
                self.model_dict[mod] = dPLHydroModel(self.config, mod).to(self.config['device'])
                self.all_model_params += list(self.model_dict[mod].parameters())

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
            self.init_optimizer()
    
    def load_model(self, model) -> None:
        model_name = str(model) + '_model_Ep' + str(self.config['epochs']) + '.pt'
        model_path = os.path.join(self.config['output_dir'], model_name)
        try:
            self.model_dict[model] = torch.load(model_path).to(self.config['device']) 
        except:
            raise FileNotFoundError(f"Model file {model_path} was not found. Check configurations.")

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
    