"""
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
"""
