"""
This is a testing script for running a dPL, physics-informed machine learning
model BMI that is NextGen framework and NOAA OWP operation-ready.

Note:
- The current setup only passes CAMELS (671 basins) data to the BMI. For
    different datasets, `.set_value()` mappings must be modeified to the respective
    forcing + attribute key values.
"""
import os
import numpy as np
from ruamel.yaml import YAML
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from dpl_bmi import BMIdPLHydroModel
from core.data.dataset_loading import get_data_dict

log = logging.getLogger(__name__)



def main() -> None:
    ################## Initialize the BMI ##################
    # Path to BMI config.
    config_path = '/data/lgl5139/hydro_multimodel/dPLHydro_multimodel/models/bmi/bmi_config.yaml' #"bmi_config.yaml"

    # Create instance of BMI model.
    log.info("Creating dPLHydro BMI model instance")
    model = BMIdPLHydroModel()

    # [CONTROL FUNCTION] Initialize the BMI.
    log.info(f"INITIALIZING BMI")
    model.initialize(bmi_cfg_filepath=config_path)


    ################## Get test data ##################
    log.info(f"Collecting attribute and forcing data")

    # TODO: Adapt this PMI data loader to be more-BMI friendly, less a function iceberg.
    dataset_dict, _ = get_data_dict(model.config, train=False)

    # Fixing typo in CAMELS dataset: 'geol_porostiy'.
    # (Written into config somewhere inside get_data_dict...)
    var_c_nn = model.config['observations']['var_c_nn']
    if 'geol_porostiy' in var_c_nn:
        model.config['observations']['var_c_nn'][var_c_nn.index('geol_porostiy')] = 'geol_porosity'


    ################## Forward model for 1 or multiple timesteps ##################
    # n_timesteps = dataset_dict['inputs_nn_scaled'].shape[0]
    # debugging ----- #
    n_timesteps = 1  # debug
    tlim = 367  # debug
    n_basins = 671
    # --------------- #

    log.info(f"BEGIN BMI FORWARD: {n_timesteps} timesteps...")

    rho = model.config['rho']  # For routing

    # TODO: implement basin batching for this loop instead of doing all basins at once.
    n_basins = 671

    # Loop through and return streamflow at each timestep.
    for t in range(n_timesteps - rho):
        # NOTE: for each timestep in this loop, the data assignments below are of
        # arrays of basins. e.g., forcings['key'].shape = (rho + 1, # basins).
        # NOTE: MHPI models use a warmup period and routing in their forward pass,
        # so we cannot simply pass one timestep to these, but rather warmup or
        # rho + 1 timesteps up to the step we want to predict.
        # TODO: Check inefficiency cost of setting an extra rho timesteps of data
        # in the BMI for each timestep prediction. If too much, we pass all available
        # data into BMI; sounds from Jonathan that this should be fine.

        ################## Map forcings + attributes into BMI ##################
        # Set NN forcings...
        for i, var in enumerate(model.config['observations']['var_t_nn']):
            standard_name = model._var_name_map_short_first[var]
            model.set_value(standard_name, dataset_dict['inputs_nn_scaled'][t:rho + t, :n_basins, i], model='nn')
        n_forc = i
        
        # Set NN attributes...
        for i, var in enumerate(model.config['observations']['var_c_nn']):
            standard_name = model._var_name_map_short_first[var]
            model.set_value(standard_name, dataset_dict['inputs_nn_scaled'][t:rho + t, :n_basins, n_forc + i + 1], model='nn') 

        # Set physics model forcings...
        for i, var in enumerate(model.config['observations']['var_t_hydro_model']):
            standard_name = model._var_name_map_short_first[var]
            model.set_value(standard_name, dataset_dict['x_hydro_model'][t:rho + t, :n_basins, i], model='pm') 

        # Set physics model attributes...
        for i, var in enumerate(model.config['observations']['var_c_hydro_model']):
            standard_name = model._var_name_map_short_first[var]
            # NOTE: These don't have a time dimension.
            model.set_value(standard_name, dataset_dict['c_hydro_model'][:n_basins, i], model='pm') 

        # [CONTROL FUNCTION] Update the model at all basins for one timestep.
        model.update()
        print(f"Streamflow at time {t} is {np.average((model.streamflow_cms))}") #.cpu()
        print(f"BMI process time: {model.bmi_process_time}")


        # flow_sim = model.get_value('land_surface_water__runoff_volume_flux')
        flow_sim = model.preds['HBV']['flow_sim']
        print(flow_sim)

    ################## Finalize BMI ##################
    # [CONTROL FUNCTION] wrap up BMI run, deallocate mem.
    log.info(f"FINALIZE BMI")
    model.finalize()



if __name__ == "__main__":
    main()
    