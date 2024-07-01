import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml
from bmipy import Bmi
from conf.config import Config
from models.model_handler import ModelHandler
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from core.data import take_sample_test

import torch
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
        self._model = None
        self._initialized = False

        self._start_time = 0.0
        self._values = {}
        self._end_time = np.finfo(float).max
        self.var_array_lengths = 1

        # Required, static attributes of the model
        _att_map = {
        'model_name':         "Hydrologic Differentiable Parameter Learning BMI",
        'version':            '1.0',
        'author_name':        'MHPI, Leo Lonzarich',
        'grid_type':          'unstructured&uniform_rectilinear',
        'time_units':         'days',
               }
        
        # TODO: Assign variables and attributes + create map (maybe in initialize with config file.)
        # Input variable names (CSDMS standard names)
        # _input_var_names = []
        self._input_forc_list = [
            'earth_surface__average_temperature'
        ]

        self._input_attr_list = [
            'basin__area'
        ]

        # Output variable names (CSDMS standard names)
        _output_var_names = []

        # TODO: will need to have this done for all desired datasets.
        # Map CSDMS Standard Names to the model's internal variable names (For GAGES-II).
        _var_name_units_map = {
            # 'land_surface_water__runoff_volume_flux':['streamflow_cms','m3 s-1'],
            # 'land_surface_water__runoff_depth':['streamflow_m','m'],
            #--------------   Dynamic inputs --------------------------------

            # 'atmosphere_water__liquid_equivalent_precipitation_rate':['total_precipitation','mm h-1'],
            ## 'atmosphere_water__liquid_equivalent_precipitation_rate':['precip', 'mm h-1'], ##### SDP
            ## 'atmosphere_water__time_integral_of_precipitation_mass_flux':['total_precipitation','mm h-1'],
            # 'land_surface_radiation~incoming~longwave__energy_flux':['longwave_radiation','W m-2'],
            # 'land_surface_radiation~incoming~shortwave__energy_flux':['shortwave_radiation','W m-2'],
            # 'atmosphere_air_water~vapor__relative_saturation':['specific_humidity','kg kg-1'],
            # 'land_surface_air__pressure':['pressure','Pa'],
            # 'land_surface_air__temperature':['temperature','degC'],
            # 'land_surface_wind__x_component_of_velocity':['wind_u','m s-1'],
            # 'land_surface_wind__y_component_of_velocity':['wind_v','m s-1'],
            #--------------   STATIC Attributes -----------------------------
            'basin__area':['area_gages2','km2'],
            'ratio__mean_potential_evapotranspiration__mean_precipitation':['aridity','-'],
            'basin__carbonate_rocks_area_fraction':['carbonate_rocks_frac','-'],
            'soil_clay__volume_fraction':['clay_frac','percent'],
            'basin__mean_of_elevation':['elev_mean','m'],
            'land_vegetation__forest_area_fraction':['frac_forest','-'],
            'atmosphere_water__precipitation_falling_as_snow_fraction':['frac_snow','-'],
            'bedrock__permeability':['geol_permeability','m2'],
            'land_vegetation__max_monthly_mean_of_green_vegetation_fraction':['gvf_max','-'],
            'land_vegetation__diff__max_min_monthly_mean_of_green_vegetation_fraction':['gvf_diff','-'],
            'atmosphere_water__mean_duration_of_high_precipitation_events':['high_prec_dur','d'],
            'atmosphere_water__frequency_of_high_precipitation_events':['high_prec_freq','d yr-1'],
            'land_vegetation__diff_max_min_monthly_mean_of_leaf-area_index':['lai_diff','-'],
            'land_vegetation__max_monthly_mean_of_leaf-area_index':['lai_max','-'],
            'atmosphere_water__low_precipitation_duration':['low_prec_dur','d'],
            'atmosphere_water__precipitation_frequency':['low_prec_freq','d yr-1'],
            'maximum_water_content':['max_water_content','m'],
            'atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate':['p_mean','mm d-1'],
            'land_surface_water__daily_mean_of_potential_evaporation_flux':['pet_mean','mm d-1'],
            'basin__mean_of_slope':['slope_mean','m km-1'],
            'soil__saturated_hydraulic_conductivity':['soil_conductivity','cm hr-1'],
            'soil_bedrock_top__depth__pelletier':['soil_depth_pelletier','m'],
            'soil_bedrock_top__depth__statsgo':['soil_depth_statsgo','m'],
            'soil__porosity':['soil_porosity','-'],
            'soil_sand__volume_fraction':['sand_frac','percent'],
            'soil_silt__volume_fraction':['silt_frac','percent'], 
            'basin_centroid__latitude':['gauge_lat', 'degrees'],
            'basin_centroid__longitude':['gauge_lon', 'degrees']
            }

        # A list of static attributes. Not all these need to be used.
        _static_attributes_list = []

    def initialize(self, bmi_cfg_filepath=None):
        """
        (BMI Control function) Initialize the dPLHydro model.

        Parameters
        ----------
        config_name : str, optional
            Name of BMI configuration file.
        """
        # Read in BMI configurations.
        if not isinstance(bmi_cfg_filepath, str) or len(bmi_cfg_filepath) == 0:
            raise RuntimeError("No valid BMI configuration provided.")

        bmi_config_file = Path(bmi_cfg_filepath).resolve()
        if not bmi_config_file.is_file():
            raise RuntimeError("No valid configuration provided.")

        with bmi_config_file.open('r') as f:
            config = yaml.safe_load(f)

        # Initialize a configuration object.
        self.bmi_config, self.bmi_config_dict = self.initialize_config(config)

        # TODO: write up maps for these.
        # These will be all the forcings and basin attributes, yeah.
        self._values = {}
        self._var_units = {}
        self._var_loc = {}
        self._grids = {}
        self._grid_type = {}

        # Set a simulation start time.
        self.current_time = self._start_time

        # Set a timstep size.
        self._time_step_size = self.bmi_config.time_step_delta

        # Initialize a trained model.
        self._model = ModelHandler(self.self.bmi_config).to(self.bmi_config.device)
        self._initialized = True

        # Intialize dataset.
        self._get_data_dict()

    def update(self):
        """
        (BMI Control function) Advance model state by one time step.
        *Note* Models should be trained standalone with dPLHydro_PMI first before forward predictions with this BMI.

        Perform all tasks that take place within one pass through the model's
        time loop.
        """
        self.current_time += self._time_step_size 
        
        self.get_tensor_slice()

        self.output = self._model.forward(self.input_tensor)

    def update_frac(self, time_frac):
        """Update model by a fraction of a time step.
        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        if self.verbose:
            print("Warning: This model is trained to make predictions on one day timesteps.")
        time_step = self.get_time_step()
        self._time_step_size = time_frac * self._time_step_size
        self.update()
        self._time_step_size = time_step

    def update_until(self, then):
        """
        (BMI Control function) Update model until a particular time.
        *Note* Models should be trained standalone with dPLHydro_PMI first before forward predictions with this BMI.

        Parameters
        ----------
        then : float
            Time to run model until.
        """
        n_steps = (then - self.get_current_time()) / self.get_time_step()

        for _ in range(int(n_steps)):
            self.update()
        self.update_frac(n_steps - int(n_steps))

    def finalize(self):
        """
        (BMI Control function) Finalize model.
        """
        # TODO: Force destruction of ESMF and other objects when testing is done
        # to save space.

        self._model = None

    def get_tensor_slice(self):
        """
        Get tensor of input data for a single timestep.
        """
        sample_dict = take_sample_test(self.bmi_config, self.dataset_dict)
        self.input_tensor = torch.Tensor()

    def _get_data_dict(self):
        from core.calc.normalize import trans_norm
        from core.utils.Dates import Dates
        from core.data.dataFrame_loading import load_data

        log.info(f"Collecting testing data")

        # Prepare training data.
        self.train_trange = Dates(self.config['train'], self.config['rho']).date_to_int()
        self.test_trange = Dates(self.config['test'], self.config['rho']).date_to_int()
        self.config['t_range'] = [self.train_trange[0], self.test_trange[1]]

        # Read data for the test time range
        dataset_dict = load_data(self.config, trange=self.test_trange)

        # Normalizatio ns
        # init_norm_stats(self.config, dataset_dict['x_nn'], dataset_dict['c_nn'], dataset_dict['obs'])
        x_nn_scaled = trans_norm(self.config, dataset_dict['x_nn'], varLst=self.config['observations']['var_t_nn'], toNorm=True)
        c_nn_scaled = trans_norm(self.config, dataset_dict['c_nn'], varLst=self.config['observations']['var_c_nn'], toNorm=True)
        c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)
        dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
        del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
        
        # Convert numpy arrays to torch tensors
        for key in dataset_dict.keys():
            if type(dataset_dict[key]) == np.ndarray:
                dataset_dict[key] = torch.from_numpy(dataset_dict[key]).float()
        self.dataset_dict = dataset_dict

        ngrid = dataset_dict['inputs_nn_scaled'].shape[1]
        self.iS = np.arange(0, ngrid, self.config['batch_basins'])
        self.iE = np.append(self.iS[1:], ngrid)




    # ------------------ Finished up to here ------------------
    # ---------------------------------------------------------
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
        for grid_id, var_name_list in self._grids.items():
            if var_name in var_name_list:
                return grid_id

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
        return len(self._model.shape)

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
        return int(np.prod(self._model.shape))

    def get_value_ptr(self, var_name):
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
        return self._values[var_name]

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

    def set_value(self, var_name, src):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        """
        val = self.get_value_ptr(var_name)
        val[:] = src.reshape(val.shape)

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
        var_name = self._grids[grid_id][0]
        shape[:] = self.get_value_ptr(var_name).shape
        return shape

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        spacing[:] = self._model.spacing
        return spacing

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        origin[:] = self._model.origin
        return origin

    def get_grid_type(self, grid_id):
        """Type of grid."""
        return self._grid_type[grid_id]

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
        return self.get_grid_size(grid)

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

    def initialize_config(cfg: DictConfig) -> Config:
        """
        Convert config into a dictionary and a Config object for validation.
        """
        try:
            config_dict: Union[Dict[str, Any], Any] = OmegaConf.to_container(
                cfg, resolve=True
            )
            config = Config(**config_dict)
        except ValidationError as e:
            log.exception(e)
            raise e
        return config, config_dict