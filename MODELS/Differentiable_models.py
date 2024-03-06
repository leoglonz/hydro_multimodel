"""
-------------------------------------
Code pulled from `PGML_STemp > HydroModels'
for demonstration purposes.

Credit Rahmani & Song et al.
-------------------------------------

Wrapper for MHPI's differentiable model type, and for the multimodel ensemble.
"""
import torch.nn

from MODELS.hydro_models.marrmot.marrmot_prms import prms_marrmot
from MODELS.hydro_models.HBV.HBVmul import HBVMul
from MODELS.hydro_models.SACSMA.SACSMAmul import SACSMAMul

from MODELS.temp_models.PGML_STemp import SNTEMP_flowSim

from MODELS.NN_models.LSTM_models import CudnnLstmModel
from MODELS.NN_models.MLP_models import MLPmul



# import MODELS
class diff_hydro_temp_model(torch.nn.Module):
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
            elif self.args["hydro_model_name"] == "HBV":
                self.hydro_model = HBVMul()
            elif self.args["hydro_model_name"] == "SACSMA":
                self.hydro_model = SACSMAMul()
            elif self.args["hydro_model_name"] != "None":
                print("hydrology (streamflow) model type has not been defined")
                exit()
            # temp_model_initialization
        if self.args["temp_model_name"] != "None":
            if self.args["temp_model_name"] == "SNTEMP":
                self.temp_model = SNTEMP_flowSim()  # this model needs a hydrology model as backbone
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
                params_dict["conv_params_temp"] = torch.sigmoid(params_temp_model[:, -4:])
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

            # Todo: send this to a function
            # source flow calculation and converting mm/day to m3/ day
            srflow, ssflow, gwflow = self.hydro_model.source_flow_calculation(self.args, flow_out,
                                                                              dataset_dictionary_sample["c_NN_sample"])
            # baseflow index percentage
            flow_out["BFI_sim"] = 100 * (torch.sum(gwflow, dim=0) / (
                    torch.sum(srflow + ssflow + gwflow, dim=0) + 0.00001))[:, 0]

            if self.args['temp_model_name'] != "None":
                # temperature model
                temp_out = self.temp_model.forward(dataset_dictionary_sample["x_temp_model_sample"],
                                                   dataset_dictionary_sample["c_temp_model_sample"],
                                                   params_dict["temp_params_raw"],
                                                   conv_params_temp=params_dict["conv_params_temp"],
                                                   args=self.args,
                                                   PET=flow_out["PET_hydro"] * (1 / (1000 * 86400)),   # converting mm/day to m/sec,
                                                   srflow=srflow,
                                                   ssflow=ssflow,
                                                   gwflow=gwflow)

                return {**flow_out, **temp_out}   # combining both dictionaries
            else:
                return flow_out
            

class hydroEnsemble(torch.nn.Module):
    # Wrapper for multiple hydrologic models.
    # In future, consider just passing the models you want to ensemble explicitly.
    def __init__(self, num_models, hidden_size, num_layers):
        super(hydroEnsemble, self).__init__()

        self.lstm = torch.nn.LSTM(num_models, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_models)  # Two models (modelA and modelB)

        # self.modelA = modelA
        # self.modelB = modelB
        # self.classifier = torch.nn.Linear(4, 2)

    def forward(self, x):
        # x is the input sequence with shape (num_basins, num_models) of concatenated model outputs.

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Fully connected layer
        fc_out = self.fc(lstm_out[:, -1, :])

        # Apply softmax activation to obtain weights
        weights = torch.nn.functional.softmax(fc_out, dim=1)

        # Weighted combination of predictions
        weighted_predictions = torch.sum(x * weights.view(-1, 1, x.size(2)), dim=1)

        # x1 = self.modelA(x1)
        # x2 = self.modelB(x2)
        # x = torch.cat((x1, x2), dim=1)
        # x = self.classifier(torch.nn.functional.softmax(x))
        # return x

        return weighted_predictions, weights
    
