"""
BMI wrapper for interfacing dPL hydrology models with NOAA OWP NextGen framework.
"""
# Need this to get external packages like conf.config.
import sys
package_path = '/data/lgl5139/hydro_multimodel/dPLHydro_multimodel'
sys.path.append(package_path)

import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict, Union

import numpy as np
import yaml
from ruamel.yaml import YAML
import torch
import time

from bmipy import Bmi
from conf.config import Config
from models.model_handler import ModelHandler
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from core.data import take_sample_test

log = logging.getLogger(__name__)



class BMIdPLHydroModel(Bmi):
    """
    Run forward with BMI for a trained differentiable hydrology model.
    """
    def __init__(self):
        """
        Create a dPLHydro model BMI ready for initialization.
        """
        super(BMIdPLHydroModel, self).__init__()
        start_time = time.time()

        self._model = None
        self._initialized = False

        self._start_time = 0.0
        self._values = {}
        self._nn_values = {}
        self._pm_values = {}
        self._end_time = np.finfo(float).max
        self.var_array_lengths = 1

        self.bmi_process_time = 0


        # Required, static attributes of the model
        _att_map = {
        'model_name':         "Differentiable Parameter Learning Hydrology BMI",
        'version':            '1.2',
        'author_name':        'MHPI',
        'time_units':         'days',
        }
        
        # Input forcing/attribute CSDMS Standard Names.
        self._input_var_names = [
            ############## Forcings ##############
            'atmosphere_water__liquid_equivalent_precipitation_rate',
            'land_surface_air__temperature',
            'land_surface_air__max_of_temperature',  # custom name
            'land_surface_air__min_of_temperature',  # custom name
            'land_surface_water__potential_evaporation_volume_flux',  # check name,
            ############## Attributes ##############
            'atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate',
            'land_surface_water__daily_mean_of_potential_evaporation_flux',
            'p_seasonality',  # custom name
            'atmosphere_water__precipitation_falling_as_snow_fraction',
            'ratio__mean_potential_evapotranspiration__mean_precipitation',
            'atmosphere_water__frequency_of_high_precipitation_events',
            'atmosphere_water__mean_duration_of_high_precipitation_events',
            'atmosphere_water__precipitation_frequency',
            'atmosphere_water__low_precipitation_duration',
            'basin__mean_of_elevation',
            'basin__mean_of_slope',
            'basin__area',
            'land_vegetation__forest_area_fraction',
            'land_vegetation__max_monthly_mean_of_leaf-area_index',
            'land_vegetation__diff_max_min_monthly_mean_of_leaf-area_index',
            'land_vegetation__max_monthly_mean_of_green_vegetation_fraction',
            'land_vegetation__diff__max_min_monthly_mean_of_green_vegetation_fraction',
            'region_state_land~covered__area_fraction',  # custom name
            'region_state_land~covered__area',  # custom name
            'root__depth',  # custom name
            'soil_bedrock_top__depth__pelletier',
            'soil_bedrock_top__depth__statsgo',
            'soil__porosity',
            'soil__saturated_hydraulic_conductivity',
            'maximum_water_content',
            'soil_sand__volume_fraction',
            'soil_silt__volume_fraction', 
            'soil_clay__volume_fraction',
            'geol_1st_class',  # custom name
            'geol_1st_class__fraction',  # custom name
            'geol_2nd_class',  # custom name
            'geol_2nd_class__fraction',  # custom name
            'basin__carbonate_rocks_area_fraction',
            'soil_active-layer__porosity',  # check name
            'bedrock__permeability'
        ]

        # Output variable names (CSDMS standard names).
        # TODO: Find CSDMS names for the other ouput vars.
        self._output_var_names = [
            'land_surface_water__runoff_volume_flux',
            'srflow',
            'ssflow',
            'gwflow',
            'AET_hydro',
            'PET_hydro',
            'flow_sim_no_rout',
            'srflow_no_rout',
            'ssflow_no_rout',
            'gwflow_no_rout',
            'excs',
            'evapfactor',
            'tosoil',
            'percolation',
            'BFI_sim'
        ]

        # Map CSDMS Standard Names to the model's internal variable names (For CAMELS).
        self._var_name_units_map = {
            ############## Forcings ##############
            'atmosphere_water__liquid_equivalent_precipitation_rate':['prcp(mm/day)', 'mm d-1'],
            'land_surface_air__temperature':['tmean(C)','degC'],
            'land_surface_air__max_of_temperature':['tmax(C)', 'degC'],  # custom name
            'land_surface_air__min_of_temperature':['tmin(C)', 'degC'],  # custom name
            'land_surface_water__potential_evaporation_volume_flux':['PET_hargreaves(mm/day)', 'mm d-1'],  # check name
            ############## Attributes ##############
            'atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate':['p_mean','mm d-1'],
            'land_surface_water__daily_mean_of_potential_evaporation_flux':['pet_mean','mm d-1'],
            'p_seasonality':['p_seasonality', '-'],  # custom name
            'atmosphere_water__precipitation_falling_as_snow_fraction':['frac_snow','-'],
            'ratio__mean_potential_evapotranspiration__mean_precipitation':['aridity','-'],
            'atmosphere_water__frequency_of_high_precipitation_events':['high_prec_freq','d yr-1'],
            'atmosphere_water__mean_duration_of_high_precipitation_events':['high_prec_dur','d'],
            'atmosphere_water__precipitation_frequency':['low_prec_freq','d yr-1'],
            'atmosphere_water__low_precipitation_duration':['low_prec_dur','d'],
            'basin__mean_of_elevation':['elev_mean','m'],
            'basin__mean_of_slope':['slope_mean','m km-1'],
            'basin__area':['area_gages2','km2'],
            'land_vegetation__forest_area_fraction':['frac_forest','-'],
            'land_vegetation__max_monthly_mean_of_leaf-area_index':['lai_max','-'],
            'land_vegetation__diff_max_min_monthly_mean_of_leaf-area_index':['lai_diff','-'],
            'land_vegetation__max_monthly_mean_of_green_vegetation_fraction':['gvf_max','-'],
            'land_vegetation__diff__max_min_monthly_mean_of_green_vegetation_fraction':['gvf_diff','-'],
            'region_state_land~covered__area_fraction':['dom_land_cover_frac', 'percent'],  # custom name
            'region_state_land~covered__area':['dom_land_cover', '-'],  # custom name
            'root__depth':['root_depth_50', '-'],  # custom name
            'soil_bedrock_top__depth__pelletier':['soil_depth_pelletier','m'],
            'soil_bedrock_top__depth__statsgo':['soil_depth_statsgo','m'],
            'soil__porosity':['soil_porosity','-'],
            'soil__saturated_hydraulic_conductivity':['soil_conductivity','cm hr-1'],
            'maximum_water_content':['max_water_content','m'],
            'soil_sand__volume_fraction':['sand_frac','percent'],
            'soil_silt__volume_fraction':['silt_frac','percent'], 
            'soil_clay__volume_fraction':['clay_frac','percent'],
            'geol_1st_class':['geol_1st_class', '-'],  # custom name
            'geol_1st_class__fraction':['glim_1st_class_frac', '-'],  # custom name
            'geol_2nd_class':['geol_2nd_class', '-'],  # custom name
            'geol_2nd_class__fraction':['glim_2nd_class_frac', '-'],  # custom name
            'basin__carbonate_rocks_area_fraction':['carbonate_rocks_frac','-'],
            'soil_active-layer__porosity':['geol_porosity', '-'],  # check name
            'bedrock__permeability':['geol_permeability','m2'],
            'drainage__area':['DRAIN_SQKM', 'km2'],  # custom name
            'land_surface__latitude':['lat','degrees'],
            ############## Outputs ##############
            # TODO: Find csdms names.
            'land_surface_water__runoff_volume_flux':['flow_sim','m3 s-1'],
            'srflow':['srflow','m3 s-1'],
            'ssflow':['ssflow','m3 s-1'],
            'gwflow':['gwflow','m3 s-1'],
            'AET_hydro':['AET_hydro','m3 s-1'],
            'PET_hydro':['PET_hydro','m3 s-1'],
            'flow_sim_no_rout':['flow_sim_no_rout','m3 s-1'],
            'srflow_no_rout':['srflow_no_rout','m3 s-1'],
            'ssflow_no_rout':['ssflow_no_rout','m3 s-1'],
            'gwflow_no_rout':['gwflow_no_rout','m3 s-1'],
            'excs':['excs','-'],
            'evapfactor':['evapfactor','-'],
            'tosoil':['tosoil','m3 s-1'],
            'percolation':['percolation','-'],
            'BFI_sim':['BFI_sim','-']
        }
        
        # Keep running total of BMI runtime.
        self.bmi_process_time += time.time() - start_time
    
    def initialize(self, bmi_cfg_filepath: Optional[str] = None) -> None:
        """
        (BMI Control function) Initialize the dPLHydro model.

        Parameters
        ----------
        bmi_cfg_filepath : str, optional
            Path to the BMI configuration file.
        """
        start_time = time.time()

        # Read in BMI configurations.
        self.initialize_config(bmi_cfg_filepath)
        
        # Make lookup tables (Peckham et al.).
        self._var_name_map_long_first = {
            long_name:self._var_name_units_map[long_name][0] for \
            long_name in self._var_name_units_map.keys()
            }
        self._var_name_map_short_first = {
            self._var_name_units_map[long_name][0]:long_name for \
            long_name in self._var_name_units_map.keys()}
        self._var_units_map = {
            long_name:self._var_name_units_map[long_name][1] for \
            long_name in self._var_name_units_map.keys()
        }

        # Initialize inputs and outputs.
        for var in self.config['observations']['var_t_nn'] + self.config['observations']['var_c_nn']:
            standard_name = self._var_name_map_short_first[var]
            self._nn_values[standard_name] = []
            # setattr(self, var, 0)

        for var in self.config['observations']['var_t_hydro_model'] + self.config['observations']['var_c_hydro_model']:
            standard_name = self._var_name_map_short_first[var]
            self._pm_values[standard_name] = []
            # setattr(self, var, 0)


        # Set a simulation start time and gettimestep size.
        self.current_time = self._start_time
        self._time_step_size = self.config['time_step_delta']

        # Load a trained model.
        self._model = ModelHandler(self.config).to(self.config['device'])
        self._initialized = True

        # Keep running total of BMI runtime.
        self.bmi_process_time += time.time() - start_time

    def update(self) -> None:
        """
        (BMI Control function) Advance model state by one time step.

        Note: Models should be trained standalone with dPLHydro_PMI first before forward predictions with this BMI.
        """
        start_time = time.time()

        self.current_time += self._time_step_size 
        
        self._values_to_dict()

        ngrid = self.dataset_dict['inputs_nn_scaled'].shape[1]
        iS = np.arange(0, ngrid, self.config['batch_basins'])
        iE = np.append(iS[1:], ngrid)
        self.dataset_sample = take_sample_test(self.config,
                                          self.dataset_dict,
                                          iS[0],
                                          iE[0]
                                          )

        # Forward dPLHydro model
        self.preds = self._model.forward(self.dataset_sample, eval=True)
        
        # Scale and check output
        self.scale_output()

    def update_frac(self, time_frac: float) -> None:
        """
        Update model by a fraction of a time step.
        
        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        if self.verbose:
            print("Warning: This model is trained to make predictions on one day timesteps.")
        time_step = self.get_time_step()
        self._time_step_size = self._time_step_size * time_frac
        self.update()
        self._time_step_size = time_step

    def update_until(self, end_time: float) -> None:
        """
        (BMI Control function) Update model until a particular time.
        Note: Models should be trained standalone with dPLHydro_PMI first before forward predictions with this BMI.

        Parameters
        ----------
        end_time : float
            Time to run model until.
        """
        n_steps = (end_time - self.get_current_time()) / self.get_time_step()

        for _ in range(int(n_steps)):
            self.update()
        self.update_frac(n_steps - int(n_steps))

    def finalize(self) -> None:
        """
        (BMI Control function) Finalize model.
        """
        # TODO: Force destruction of ESMF and other objects when testing is done
        # to save space.
        
        torch.cuda.empty_cache()
        self._model = None

    def array_to_tensor(self) -> None:
        """
        Converts input values into Torch tensor object to be read by model. 
        """  
        raise NotImplementedError("array_to_tensor")
    
    def tensor_to_array(self) -> None:
        """
        Converts model output Torch tensor into date + gradient arrays to be
        passed out of BMI for backpropagation, loss, optimizer tuning.
        """  
        raise NotImplementedError("tensor_to_array")
    
    def get_tensor_slice(self):
        """
        Get tensor of input data for a single timestep.
        """
        # sample_dict = take_sample_test(self.bmi_config, self.dataset_dict)
        # self.input_tensor = torch.Tensor()
    
        raise NotImplementedError("get_tensor_slice")

    def get_var_type(self, var_name):
        """
        Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ptr(var_name).dtype)

    def get_var_units(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Variable units.
        """
        return self._var_units[var_name]

    def get_var_nbytes(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    def get_var_itemsize(self, name):
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name):
        return self._var_loc[name]

    def get_var_grid(self, var_name):
        """Grid id for a variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Grid id.
        """
        # for grid_id, var_name_list in self._grids.items():
        #     if var_name in var_name_list:
        #         return grid_id
        raise NotImplementedError("get_var_grid")

    def get_grid_rank(self, grid_id):
        """Rank of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Rank of grid.
        """
        # return len(self._model.shape)
        raise NotImplementedError("get_grid_rank")

    def get_grid_size(self, grid_id):
        """Size of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Size of grid.
        """
        # return int(np.prod(self._model.shape))
        raise NotImplementedError("get_grid_size")

    def get_value_ptr(self, var_name: str, model:str) -> np.ndarray:
        """Reference to values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Value array.
        """
        if model == 'nn':
            if var_name not in self._nn_values.keys():
                raise ValueError(f"No known variable in BMI model: {var_name}")
            return self._nn_values[var_name]

        elif model == 'pm':
            if var_name not in self._pm_values.keys():
                raise ValueError(f"No known variable in BMI model: {var_name}")
            return self._pm_values[var_name]
        
        else:
            raise ValueError("Valid model type (nn or pm) must be specified.")

    def get_value(self, var_name, dest):
        """Copy of values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.

        Returns
        -------
        array_like
            Copy of values.
        """
        dest[:] = self.get_value_ptr(var_name).flatten()
        return dest

    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.

        Returns
        -------
        array_like
            Values at indices.
        """
        dest[:] = self.get_value_ptr(var_name).take(indices)
        return dest

    def set_value(self, var_name, values: np.ndarray, model:str):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        values : array_like
            Array of new values.
        """
        if not isinstance(values, (np.ndarray, list, tuple)):
            values = np.array([values])

        val = self.get_value_ptr(var_name, model=model)

        # val = values.reshape(val.shape)
        val[:] = values

    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ptr(name)
        val.flat[inds] = src

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    def get_input_var_names(self):
        """Get names of input variables."""
        return self._input_var_names

    def get_output_var_names(self):
        """Get names of output variables."""
        return self._output_var_names

    def get_grid_shape(self, grid_id, shape):
        """Number of rows and columns of uniform rectilinear grid."""
        # var_name = self._grids[grid_id][0]
        # shape[:] = self.get_value_ptr(var_name).shape
        # return shape
        raise NotImplementedError("get_grid_shape")

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        # spacing[:] = self._model.spacing
        # return spacing
        raise NotImplementedError("get_grid_spacing")

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        # origin[:] = self._model.origin
        # return origin
        raise NotImplementedError("get_grid_origin")

    def get_grid_type(self, grid_id):
        """Type of grid."""
        # return self._grid_type[grid_id]
        raise NotImplementedError("get_grid_type")

    def get_start_time(self):
        """Start time of model."""
        return self._start_time

    def get_end_time(self):
        """End time of model."""
        return self._end_time

    def get_current_time(self):
        return self._current_time

    def get_time_step(self):
        return self._time_step_size

    def get_time_units(self):
        return self._time_units

    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid):
        """Number of grid nodes.

        Parameters
        ----------
        grid : int
            Identifier of a grid.

        Returns
        -------
        int
            Size of grid.
        """
        # return self.get_grid_size(grid)
        raise NotImplementedError("get_grid_node_count")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_x(self, grid, x):
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid, y):
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid, z):
        raise NotImplementedError("get_grid_z")

    def initialize_config(self, config_path: str) -> Dict:
        """
        Check that config_path is valid path and convert config into a
        dictionary object.
        """
        config_path = Path(config_path).resolve()
        
        if not config_path:
            raise RuntimeError("No BMI configuration path provided.")
        elif not config_path.is_file():
            raise RuntimeError(f"BMI configuration not found at path {config_path}.")
        else:
            with config_path.open('r') as f:
                self.config = yaml.safe_load(f)
    
        # USE BELOW FOR HYDRA + OMEGACONF:
        # try:
        #     config_dict: Union[Dict[str, Any], Any] = OmegaConf.to_container(
        #         cfg, resolve=True
        #     )
        #     config = Config(**config_dict)
        # except ValidationError as e:
        #     log.exception(e)
        #     raise e
        # return config, config_dict

    def scale_output(self) -> None:
        """
        Scale and return more meaningful output from wrapped model.
        """
        models = self.config['hydro_models'][0]

        # TODO: still have to finish finding and undoing scaling applied before
        # model run. (See some checks used in bmi_lstm.py.)

        # Strip unnecessary time and variable dims. This gives 1D array of flow
        # at each basin.
        # TODO: setup properly for multiple models later.
        self.streamflow_cms = self.preds[models]['flow_sim'].squeeze()

    def take_sample_test(self, config: Dict, dataset_dictionary: Dict[str, torch.Tensor], 
                        i_s: int, i_e: int) -> Dict[str, torch.Tensor]:
        """
        Take sample of data for testing batch.
        """
        dataset_sample = {}
        for key, value in dataset_dictionary.items():
            if value.ndim == 3:
                # TODO: I don't think we actually need this.
                # Remove the warmup period for all except airtemp_memory and hydro inputs.
                if key in ['airT_mem_temp_model', 'x_hydro_model', 'inputs_nn_scaled']:
                    warm_up = 0
                else:
                    warm_up = config['warm_up']
                dataset_sample[key] = value[warm_up:, i_s:i_e, :].to(config['device'])
            elif value.ndim == 2:
                dataset_sample[key] = value[i_s:i_e, :].to(config['device'])
            else:
                raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")
        return dataset_sample

    def _values_to_dict(self) -> None:
        """
        Take CSDMS Standard Name-mapped forcings + attributes and construct data
        dictionary for NN and physics model.
        """
        # n_basins = self.config['batch_basins']
        n_basins = 671
        rho = self.config['rho']

        # Initialize dict arrays
        x_nn = np.zeros((rho + 1, n_basins, len(self.config['observations']['var_t_nn'])))
        c_nn = np.zeros((rho + 1, n_basins, len(self.config['observations']['var_c_nn'])))
        x_hydro_model = np.zeros((rho + 1, n_basins, len(self.config['observations']['var_t_hydro_model'])))
        c_hydro_model = np.zeros((n_basins, len(self.config['observations']['var_c_hydro_model'])))

        for i, var in enumerate(self.config['observations']['var_t_nn']):
            standard_name = self._var_name_map_short_first[var]
            # NOTE: Using _values is a bit hacky. Should use get_values I think.    
            x_nn[:, :, i] = np.array([self._nn_values[standard_name]])
        
        for i, var in enumerate(self.config['observations']['var_c_nn']):
            standard_name = self._var_name_map_short_first[var]
            c_nn[:, :, i] = np.array([self._nn_values[standard_name]])

        for i, var in enumerate(self.config['observations']['var_t_hydro_model']):
            standard_name = self._var_name_map_short_first[var]
            x_hydro_model[:, :, i] = np.array([self._pm_values[standard_name]])

        for i, var in enumerate(self.config['observations']['var_c_hydro_model']):
            standard_name = self._var_name_map_short_first[var]
            c_hydro_model[:, i] = np.array([self._pm_values[standard_name]])
        
        self.dataset_dict = {
            'inputs_nn_scaled': np.concatenate((x_nn, c_nn), axis=2), #[np.newaxis,:,:],
            'x_hydro_model': x_hydro_model, #[np.newaxis,:,:],
            'c_hydro_model': c_hydro_model
        }
        print(self.dataset_dict['inputs_nn_scaled'].shape)

        # Convert to torch tensors:
        for key in self.dataset_dict.keys():
            if type(self.dataset_dict[key]) == np.ndarray:
                self.dataset_dict[key] = torch.from_numpy(self.dataset_dict[key]).float() #.to(self.config['device'])