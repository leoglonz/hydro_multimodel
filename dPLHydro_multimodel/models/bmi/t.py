
class BMIdPLHydroModel(Bmi):
    def __init__(self):
        # Do we even need all of this? why cant it all be packaged in array?
        self._input_forc_list = [
            'earth_surface__average_temperature'
        ]

        self._attr_list = [
            'basin__area'
        ]

        self._standard_var_map = {
            # Forcings
            'earth_surface__average_temperature': ['tmean(C)','degC'],
'atmosphere_water__one-day_time_integral_of_precipitation_leq-volume_flux': ['prcp(mm/day)', 'mm-day']
            # Attributes
            'basin__area':['area_gages2','km2'],
        }

    def initialize(self, bmi_cfg_filepath=None):

        # Initialize forcings and attributes so that they can be set externally
        # When passing data into BMI.
        for key in list(self._standard_var_map.keys()):
            if key in self._input_forc_names.keys(): 
                self._forcings[key] = 0
            elif key in self._attr_list.keys():
                self._attrs[key] = 0



#### Executing model:

model = BMIdPLHydroModel()

model.initialize()

# Set forcings...
model.setvalue('earth_surface__average_temperature', forcings['tmean(C)'])

# Set attributes...
model.setvalue('basin__area', attributes['area_gages2'])

model.update()

# etc.















    # # TODO: write a timestep handler/translator so we can control we can pull out
    # # forcings/attributes for the specific timesteps we want streamflow predictions for.

    # # Loop through and return streamflow at each timestep.
    # for t in range(n_forcings):
    #     # NOTE: for each timestep in this loop, the data assignments below are of
    #     # arrays of basins. e.g., forcings['key'].shape = (1, # basins)

    #     # Set CAMELS forcings...
    #     model.setvalue('atmosphere_water__liquid_equivalent_precipitation_rate', forcings['prcp(mm/day)'])
    #     model.setvalue('land_surface_air__temperature', forcings['tmean(C)'])
    #     model.setvalue('land_surface_water__potential_evaporation_volume_flux', forcings['PET_hargreaves(mm/day)'])  # confirm correct name

    #     # Set CAMELS attributes...
    #     model.setvalue('atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate', attributes['p_mean'])
    #     model.setvalue('land_surface_water__daily_mean_of_potential_evaporation_flux', attributes['pet_mean'])
    #     model.setvalue('p_seasonality', attributes['p_seasonality'])  # custom name
    #     model.setvalue('atmosphere_water__precipitation_falling_as_snow_fraction', attributes['frac_snow'])
    #     model.setvalue('ratio__mean_potential_evapotranspiration__mean_precipitation', attributes['aridity'])
    #     model.setvalue('atmosphere_water__frequency_of_high_precipitation_events', attributes['high_prec_freq'])
    #     model.setvalue('atmosphere_water__mean_duration_of_high_precipitation_events', attributes['high_prec_dur'])
    #     model.setvalue('atmosphere_water__precipitation_frequency', attributes['low_prec_freq'])
    #     model.setvalue('atmosphere_water__low_precipitation_duration', attributes['low_prec_dur'])
    #     model.setvalue('basin__mean_of_elevation', attributes['elev_mean'])
    #     model.setvalue('basin__mean_of_slope', attributes['slope_mean'])
    #     model.setvalue('basin__area', attributes['area_gages2'])
    #     model.setvalue('land_vegetation__forest_area_fraction', attributes['frac_forest'])
    #     model.setvalue('land_vegetation__max_monthly_mean_of_leaf-area_index', attributes['lai_max'])
    #     model.setvalue('land_vegetation__diff_max_min_monthly_mean_of_leaf-area_index', attributes['lai_diff'])
    #     model.setvalue('land_vegetation__max_monthly_mean_of_green_vegetation_fraction', attributes['gvf_max'])
    #     model.setvalue('land_vegetation__diff__max_min_monthly_mean_of_green_vegetation_fraction', attributes['gvf_diff'])
    #     model.setvalue('region_state_land~covered__area_fraction', attributes['dom_land_cover_frac'])  # custom name
    #     model.setvalue('region_state_land~covered__area', attributes['dom_land_cover'])  # custom name
    #     model.setvalue('root__depth', attributes['root_depth_50'])  # custom name
    #     model.setvalue('soil_bedrock_top__depth__pelletier', attributes['soil_depth_pelletier'])
    #     model.setvalue('soil_bedrock_top__depth__statsgo', attributes['soil_depth_statsgo'])
    #     model.setvalue('soil__porosity', attributes['soil_porosity'])
    #     model.setvalue('soil__saturated_hydraulic_conductivity', attributes['soil_conductivity'])
    #     model.setvalue('maximum_water_content', attributes['max_water_content'])
    #     model.setvalue('soil_sand__volume_fraction', attributes['sand_frac'])
    #     model.setvalue('soil_silt__volume_fraction', attributes['silt_frac'])
    #     model.setvalue('soil_clay__volume_fraction', attributes['clay_frac'])
    #     model.setvalue('geol_1st_class', attributes['geol_1st_class'])  # custom name
    #     model.setvalue('geol_1st_class__fraction', attributes['glim_1st_class_frac'])  # custom name
    #     model.setvalue('geol_2nd_class', attributes['geol_2nd_class'])  # custom name
    #     model.setvalue('geol_2nd_class__fraction', attributes['glim_2nd_class_frac'])  # custom name
    #     model.setvalue('basin__carbonate_rocks_area_fraction', attributes['carbonate_rocks_frac'])  # custom name
    #     model.setvalue('soil_active-layer__porosity', attributes['geol_porosity'])  # confirm correct name
    #     model.setvalue('bedrock__permeability', attributes['geol_permeability'])

    #     # [CONTROL FUNCTION] Update the model at all basins for one timestep.
    #     model.update()
    #     print(f"Streamflow at time {model.t} is {model.streamflow_cms}")