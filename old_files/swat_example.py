import pickle

train_file = 'training_file'
validation_file = 'validation_file'
# Load X, Y, C from a file
with open(train_file, 'rb') as f:
    train_x, train_y, train_c = pickle.load(f)  # Adjust this line based on your file format
with open(validation_file, 'rb') as g:
    val_x, val_y, val_c = pickle.load(g)

import math
from datetime import datetime, timedelta

import numpy as np
import swat_functional as F
#------NEW------------
import torch
import torch.nn as nn

# global definitions:
device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32
def createTensor(dims, requires_grad=False):
  # a small function to centrally manage device, data types, etc., of new arrays
  return torch.zeros(dims,requires_grad=requires_grad,dtype=dtype).to(device)

def createDictFromKeys(keyList, mtd=0, dims=None, dat=None):
  d = {}
  for kk, k in enumerate(keyList):
    if mtd == 0 or mtd is None or mtd == "None":
      d[k] = None
    elif mtd == 1 or mtd == 'zeros':
      d[k] = createTensor(dims)
    elif mtd == 11 or mtd == 'value':
      d[k] = createTensor(dims) + dat
    elif mtd == 2 or mtd == 'ref':
      d[k] = dat[..., kk]
    elif mtd == 21 or mtd == 'refClone':
      d[k] = dat[..., kk].clone()
  return d

def scaleDict(pDict, pKeys, rangeDict):
  """
  Scale the values in a dict based on a given range.
  """
  for k in pKeys:
    pDict[k] = pDict[k] * (rangeDict[k][1] - rangeDict[k][0]) + rangeDict[k][0]
  return pDict

class multiphysModel(nn.Module):
  # a wrapper that aggregates models and forcings and controls the weights at the metamodel level
  # to drive a collection of models or a collection of forcings or simulations
  def __init__(self, modelDict, paramSettings):
    super(multiphysModel, self).__init__()
    self.modelsDict = modelDict
    #self.modelsDict = modelDict

  def initModel(self, key, *args, **kwargs):
    # this one has to be explicit right now
    self.modelsDict[0] = SWAT(*args, **kwargs)

  def forward(self, *args, **kwargs):
    for key in self.modelsDict:
      self.modelDict[key](*args, **kwargs)



class SWAT(nn.Module):
  def __init__(self, settings={"nz":5, 'applyParams':None}):
    # here we initialize storages permanent across minibatches, such as embedded (process) NN weights
    # parameterization NN weights could be held here or stored up a level
    # otherwise just initialize the name
    super(SWAT, self).__init__()
    self.name = 'SWAT'
    self.settings = settings
    if settings['applyParams'] is None:
        self.applyParams = self.applyParams_default
    else:
        self.applyParams = settings['applyParams']

    # default keys
    self.defaultKeys = {}
    self.defaultRangesDict = {}
    self.defaultKeys['forcingsTS'] = ('precipday', 'tmx', 'tmn', 'tmp_an', 'tmpav', 'hru_rad', 'laiday', 'iida', 'rhd')
    self.defaultKeys['attributes'] = ('fcimp', 'iurban', 'urblu', 'sol_nly','hru_slp', 'slsoil', 'dep_imp',
                                      'tloss', 'twlpond', 'twlwet', 'gwq_ru', 'sub_elev', 'lat',
                                      'cnn', 'sol_alb', 'surlag', 'tconc', 'sub_sftmp', 'rad_factor', 'sub_sftmp',
                                      'sub_smfmx', 'sub_smfmn', 'sub_smtmp', 'sub_timp', 'snocovmx', 'snocov1',
                                      'snocov2', 'ffc', 'alpha_bf', 'gw_delaye', 'gw_revap', 'gw_spyld', 'gwqmn',
                                      'rcheg_dp', 'revapmn', 'gwqmn', 'alpha_bfe_d', 'harg_petco', 'esco')
    self.defaultKeys['states'] = ('bio_ms', 'sol_rsd', 'sno_hru', 'surf_bs','sol_cov',
                                  'deepst', 'gw_q', 'gwht', 'rchrg', 'sepbtm',
                                  'shallst', 'rchrg_src', 'wtab')
    self.defaultKeys['fluxes'] = ('surfq', 'qday', 'gwseep', 'revapday', 'pet_day',
                                  'gw_qdeep', 'rhd', 'esday', 'inflpcp')
    self.defaultKeys['vars'] = ('cnday', 'sol_avbd', 'sol_sumfc', 'sol_sumul', 'sci', 'smx', 'wrt1', 'wrt2', 'brt', 'albday',
    'sol_fc', 'sol_st', 'sol_ul')
    self.defaultKeys['states2D'] = ('sol_sw', 'sol_tmp')
    self.defaultKeys['attributes2D'] = ('sol_z','sol_clay', 'sol_silt', 'sol_bd', 'sol_k', 'sol_mwc', 'awc_factor')
    self.defaultRangesDict['attributes'] = {'surlag': [1, 30], 'cnn':[30,100], 'sol_alb':[0.05, 0.20],
                             'tconc': [0.5, 6], 'sub_sftmp':[-2,2], 'rad_factor':[0.1, 0.99], 'sub_smfmx':[3, 20],
                             'sub_smfmn':[1,10], 'sub_smtmp':[0,2], 'sub_timp':[0,1], 'snocovmx':[10, 100],
                             'snocov1':[1, 10], 'snocov2':[1,10], 'ffc':[0, 0.6], 'alpha_bf':[0.001, 0.99], 'gw_delaye':[0.001, 0.99],
                             'gw_revap':[0.001, 0.99], 'gw_spyld':[0.01, 0.35], 'gwqmn':[10, 200], 'rchrg_dep':[0,1],
                             'rchrg_dp':[0,0.4],'revapmn':[20, 200], 'alpha_bfe_d':[0.38, 0.99], 'harg_petco':[0.0019, 0.0032], 'esco':[0,1],
                             'awc_factor':[0.01, 0.99]} # used to scale parameters to this range.

  def applyParams_default(self, pDict):
      self.pPhysDict = scaleDict(pDict, pDict.keys(), self.defaultRangesDict['attributes'])
      self.attributes.update(self.pPhysDict)
      self.attributes2D.update(self.pPhysDict)
      # this is a default method to change parameters
      # directly copy the content in pDict into attributes. Will retain all other fields originally in self.attributes
      # the dimensions must be correct in pDict

  def getForcingDay(self, iday):
      for k in self.forcings:
          if self.forcingsTS[k].ndim==2:
              self.forcings[k] = self.forcingsTS[k][:, iday]
          elif  self.forcingsTS[k].ndim==3:
              self.forcings[k] = self.forcingsTS[k][:, iday, :]

  def preRun(self, x, c):
    # pytorch-version-specific initializations
    # here we initialize the data structures and variables which are fresh for every minibatch
    A,A2 = c
    nbasin,nt,nx= x.shape; nhru,na = A.shape; ncomp=nhru/nbasin
    # for debugging, assuming x is also of nhru,nt,nx. In fact it should not be. it should be nbasin, nt, nx. will do this later.
    self.dims = {"nhru": nhru, "nbasin": nbasin, "nz": self.settings["nz"], "ncomp": ncomp, "nt": nt, "np": np}

    # in dHBV terminology, nhru could be nbasin*ncomponent
    # all data in here have HRU dimension.
    # each call creates a dict with initial value initialization
    # we divide arrays into several types:
    # states: which carry the memory of the physical system and exist throughout the simulation. They needs initial values (could be 0+warm up).
    # The states, forcing, attributes and parameters completely determine how the model evolves to the next step.
    # states2D: states with a vertical discretization
    # fluxes: flows that transfer mass/energy/materials between states.
    # vars: intermediate derived variables that need to be referred to across the model. Difference from states is whether it is completely determined by states
    # NO local variables in this part
    # attributes: features (likely direct derived from GIS data) that are fixed and not likely changed
    # parameters: features considered tunable physical parameters

    # link input data and create fresh states
    dKeys = self.defaultKeys
    self.forcingsTS = createDictFromKeys(dKeys['forcingsTS'], mtd='ref', dat=x)
    self.forcings = createDictFromKeys(self.forcingsTS.keys(), mtd=None)
    self.attributes = createDictFromKeys(dKeys['attributes'], mtd='ref', dat=A)
    self.states = createDictFromKeys(dKeys['states'], dims=self.dims["nhru"], mtd='zeros')
    dim2D = (self.dims["nhru"], self.dims["nz"])
    self.states2D = createDictFromKeys(dKeys['states2D'], mtd='zeros', dims=dim2D)
    self.attributes2D = createDictFromKeys(dKeys['attributes2D'], mtd='ref', dat=A2)
    self.fluxes = createDictFromKeys(dKeys['fluxes'], mtd=None)
    self.vars = createDictFromKeys(dKeys['vars'], mtd=None)

    # adhoc initialization
    self.initAttributes()
    self.initAdhoc()

    # this is from swat
    self.readinpt()

  def initAdhoc(self):
    self.states['bio_ms'][:] = 1.0 #need better value
    self.states['sol_rsd'][:] = 1.0 #need better value

    #self.states['sno_hru'][:] = 0
    #self.states['surf_bs'][:] = 0

    #self.states2D['sol_sw'][:] = 0
    self.states2D['sol_tmp'][:] = 15 #average soil temperature


  def initAttributes(self): # adhoc code for those values that need to be assumed.
    #self.attributes['fcimp'][:] = 0.0
    self.attributes['iurban'][:] = 0.0
    self.attributes['urblu'][:] = 0.0

    self.attributes['sol_nly'][:] = 5
    self.attributes['ffc'][:] = 0.2

  def surfst_h2o(self):
    self.fluxes['qday'], self.states['surf_bs'] = F.surfst_h2o(self.vars['brt'], self.states['surf_bs'], self.fluxes['surfq'])

  def surq_daycn(self):
    self.fluxes['surfq'] = F.surq_daycn(self.vars['cnday'], self.attributes['fcimp'], self.attributes['iurban'],
                                        self.forcings['precipday'], self.attributes['urblu'])

  def dailycn(self):
    self.states['cnday'] = F.dailycn(self.vars['sci'], self.vars['smx'], self.stats2D['sol_sw'],
                                     self.stats2d['sol_tmp'], [self.vars['wrt1'], self.vars['wrt2']] , 0, 0) #icn:0, cn_froz=0

  def curno(self):
    _, __, ___, self.vars['smx'], self.vars['sci'], self.vars['wrt1'], self.vars['wrt2'] = F.curno(self.attributes['cnn'],
                                                                               self.vars['sol_sumfc'], self.vars['sol_sumul'])

  def solt(self):
    self.states2D['sol_tmp'] = F.solt(self.vars['albday'], self.forcings['hru_rad'], self.attributes['rad_factor'], self.states['sno_hru'],
                                      self.vars['sol_avbd'], self.states['sol_cov'], self.attributes['sol_nly'], self.states2D['sol_sw'],
                                      self.states2D['sol_tmp'], self.attributes2D['sol_z'], self.forcings['tmn'], self.forcings['tmp_an'],
                                      self.forcings['tmpav'], self.forcings['tmx'])

  def albedo(self):
    self.vars['albday'] = F.albedo(self.states['sol_cov'], self.attributes['sol_alb'], self.forcingsTS['laiday'], self.states['sno_hru'])

  def hydroinit(self):
    self.vars['brt'] = F.hydroinit(self.attributes['surlag'], self.attributes['tconc'])

  def snom(self):
    self.states['sno_hru'], _, __, ___ = F.snom_no_elevation_bands(self.states['sno_hru'], self.states2D['snotmp'], self.forcings['tmpav'],
                                                       self.forcings['tmx'], self.forcings['precipday'], self.attributes['sub_sftmp'], self.attributes['sub_smfmx'],
                                                                   self.attributes['sub_smfmn'], self.attributes['sub_smtmp'], self.attributes['sub_timp'], self.forcings['iida'],
                                                                   self.attributes['snocovmx'], self.attributes['snocov1'], self.attributes['snocov2'])

  def plantmod(self):
    self.states['sol_cov'] = F.plantmod(self.states['bio_ms'], self.states['sol_rsd'])

  def soil_phys(self):
    self.states2D['sol_sw'], self.vars['sol_sumfc'], self.vars['sol_sumul'], self.vars['sol_avbd'], self.vars['sol_fc'], self.vars['sol_ul'] = \
      F.soil_phys(self.attributes2D['sol_mwc'], self.attributes2D['awc_factor'], self.attributes2D['sol_clay'], self.attributes['ffc'], self.attributes2D['sol_bd'],
                                          self.attributes2D['sol_z'])

  def percmain(self):
    self.states2D['sol_sw'] = F.percmain(self.fluxes['inflpcp'], self.states2D['sol_st'], self.vars['sol_fc'], self.attributes2D['sol_z'],
                                         self.vars['sol_ul'], self.attributes2D['sol_k'], self.attributes['hru_slp'], self.attributes['slsoil'],
                                         self.states2D['sol_tmp'], self.attributes['sol_nly'], self.attributes['dep_imp'], self.states2D['sol_sw'])


  def gwmod(self):
    self.states['deepst'], self.states['gw_q'], self.states['gwht'],
    self.fluxex['gwseep'],self.states['rchrg'], self.fluxes['revapday'], self.states['shallst'] = F.gwmod(self.attributes['alpha_bf'], self.states['gw_q'],
                                                                                    self.states['deepst'], self.attributes['gw_delaye'],
                                                                                    self.attributes['gw_revap'], self.attributes['gw_spyld'],
                                                                                    self.states['gwht'], self.attributes['gwqmn'],
                                                                                    self.fluxex['pet_day'], self.states['rchrg'],
                                                                                    self.attributes['rchrg_dp'], self.attributes['revapmn'],
                                                                                    self.states['sepbtm'], self.states['shallst'],
                                                                                    self.attributes['tloss'], self.attributes['twlpond'],
                                                                                    self.attributes['twlwet'], self.attributes['gwq_ru'],
                                                                                    self.states['rchrg_src'])
  def gwmod_deep(self):
    self.states['deepst'], self.fluxes['gw_qdeep'] = F.gwmod_deep(self.attributes['alpha_bfe_d'], self.states['deepst'],
                                                                self.fluxes['gwseep'], self.states['shallst'],
                                                                self.attributes['gwqmn'])
  def wattable(self):
    self.states['wtab'] = F.wattable(self.forcings['precipday'], self.fluxes['qday'], self.fluxes['pet_day'], self.states['wtab'])

  def etpot(self):
    #ipet=0: taken as priestly-taylor method default (rhd to figure out, right now generated from NN)
    self.fluxes['pet_day'] = F.etpot(0, self.forcings['tmpav'], self.attributes['sub_elev'], self.forcings['rhd'], self.states['sno_hru'],
                                   self.forcings['hru_rad'], self.attributes['rad_factor'], self.forcings['iida'], self.attributes['lat'],
                                   self.attributes['harg_petco'], self.forcings['tmx'], self.forcings['tmn'])

  def etact(self):
    self.fluxes['es_day'], self.states2D['sol_sw'] = F.etact(self.fluxes['pet_day'], self.forcings['laiday'], self.states['sno_hru'],
                                                         self.states['sol_cov'], self.forcings['tmpav'], self.attributes2D['sol_z'],
                                                         self.vars['sol_st'], self.vars['sol_fc'], self.attributes['esco'], self.states2D['sol_sw'])


  def readinpt(self):
    # mimic swat initiation sequence
    self.soil_phys()
    self.hydroinit()

  def subbasin(self):
    # ignoring water_hru
    #phubase(j) = phubase(j) + tmpav(j) / phutot(hru_sub(j))
    self.albedo()
    self.solt()
    self.etpot()
    self.etact()
    self.surface()

    #!! compute effective rainfall (amount that percs into soil)
	#lid_str_curday(j,:) = lid_str_curday(j,:) / (hru_ha(j) * 10.) !m3 to mm
	#lid_sto = sum(lid_str_curday(j,:))
	#inflpcp = Max(0.,precipday - surfq(j) - lid_sto)
    self.fluxes["inflpcp"] = self.forcings["precipday"] -  self.fluxes["surfq"]

    self.percmain()
    self.wattable()
    # ignoring confert
    # ignoring conapply
    # ignoring graze
    self.plantmod()
    # ignoring etday = ep_day + es_day + canev
    # ignoring call nminrl
    # ignoring call carbon
    # ignoring call nitvol
    # ignoring call pminrl
    # ignoring call biozone
    self.gwmod()
    self.gwmod_deep()
    # ignoring call washp
    # ignoring call decay
    # ignoring call pestlch
    # ignoring call enrsb(0)
    # ignoring call pesty(0)
    # ignoring call orgn(0)
    # ignoring call orgncswat(0)
    # ignoring call psed(0)
    # ignoring call nrain
    # ignoring call nlch
    # ignoring call solp
    # ignoring call bacteria
    # ignoring call urban
    # ignoring call pothole
    # ignoring call latsed
    # ignoring call gwnutr
    # ignoring call gw_no3
    # ignoring call surfstor # maybe needed
    # ignoring call substor # maybe needed
    # ignoring call grass_wway
    # ignoring call bmpfixed
    # ignoring call wetlan
    # ignoring call hrupond
    # ignoring call pothole
    # ignoring call watuse
    # ignoring call watbal # maybe needed
    # ignoring call subwq
    # ignoring call sumv
    # ignoring call virtual
    # ignoring call routels(iru_sub) # maybe needed
    # ignoring call sumhyd # maybe needed
    # ignoring call addh # maybe needed
    # ignoring call sumhyd # maybe needed
    # ignoring varoute(isub,:) = varoute(ihout,:)

  def surface(self):
    # canopyint()
    self.snom()
    # ignoring overland flow from upstream routing unit
    # ignoring irrigation from retention-irrigation ponds to soil water
    #calculate subdaily curve number value
    self.dailycn()
    # ignoring BMP adjustment
    # ignoring crack flow
    #compute runoff - surfq in mm H2O
    self.surq_daycn()
    # ignoring Water stored in urban LIDs drains if no rainfall occurs for 4 days. Jaehak 2021
    # ignoring !add irrigation from cistern to top soil moisture content
    # ignoring surfq(j) = surfq(j) + qird
    # !! calculate amount of surface runoff reaching main channel during day
    self.surfst_h2o()
    # if (precipday > 0.01) call alph(0)
    self.pkq()
    # ignoring !! compute transmission losses for non-HUMUS datasets
    # ignoring !! calculate sediment erosion by rainfall and overland flow
    # ignoring !! cfactor

  def forward(self, x, A, p=None):
    # from outside model
    self.preRun(x, A)
    if p is not None: self.applyParams_default(p)

    nt = self.dims["nz"]
    for i in range(nt):
      self.getForcingDay(i)
      self.subbasin()

def julian_date(start_date, end_date):
    """
    Generate a list of days of the year for each day from start_date to end_date.
    :param start_date: Start date as an integer in 'YYYYMMDD' format.
    :param end_date: End date as an integer in 'YYYYMMDD' format.
    :return: List of days of the year.
    """
    start_year = start_date // 10000
    start_month = (start_date % 10000) // 100
    start_day = start_date % 100
    end_year = end_date // 10000
    end_month = (end_date % 10000) // 100
    end_day = end_date % 100

    start_datetime = datetime(start_year, start_month, start_day)
    end_datetime = datetime(end_year, end_month, end_day)

    day_list = []
    current_day = start_datetime
    while current_day <= end_datetime:
        day_list.append(current_day.timetuple().tm_yday)
        current_day += timedelta(days=1)

    return day_list[:-1]


def calculate_rh(temperature, vapor_pressure_pa):
    """
    Calculate the relative humidity based on temperature and vapor pressure in Pascals.

    Parameters:
    temperature (float): Temperature in degrees Celsius.
    vapor_pressure_pa (float): Actual vapor pressure in Pascals (Pa).

    Returns:
    float: Relative Humidity in percentage.
    """
    # Convert vapor pressure from Pa to hPa
    vapor_pressure_hpa = vapor_pressure_pa / 100

    # Calculate saturation vapor pressure using Tetens formula (in hPa)
    saturation_vapor_pressure = 6.112 * np.exp(17.67 * temperature / (temperature + 243.5))

    # Calculate relative humidity
    relative_humidity = (vapor_pressure_hpa / saturation_vapor_pressure)

    return relative_humidity



def calculate_laiday(lai_max, lai_diff, day_of_year, phase_shift=0):
    period = 365.25  # Period of the cycle, accounting for leap years
    # Calculate laiday using the sinusoidal function
    # day_of_year = np.array(list(map(float, day_of_year)))
    laiday = lai_max - (lai_diff / 2) * np.sin((2 * np.pi * day_of_year) / period + phase_shift)
    return laiday


def calculate_slope_length(mean_slope_m_per_km, distance_km):
    # Convert the slope to meters per meter (or rise over run)
    mean_slope = mean_slope_m_per_km / 1000  # Convert m/km to m/m

    # Calculate elevation change
    elevation_change = mean_slope * distance_km * 1000  # Convert distance back to meters

    # Calculate the slope length
    slsoil = math.sqrt((distance_km * 1000)**2 + elevation_change**2)
    return slsoil
#slsoil = 50m

import pandas as pd


def an_temp(tmp_arr, Ttrain):
    start_date = pd.to_datetime(str(Ttrain[0]), format='%Y%m%d')
    end_date = pd.to_datetime(str(Ttrain[1]), format='%Y%m%d')
    dates = pd.date_range(start=start_date, end=end_date)
    temp_df = pd.DataFrame(tmp_arr, index=dates[:-1])
    temp_df.columns = ['Temp']
    temp_df['year'] = temp_df.index.year

    # yearly_mean = temp_df['Temp'].resample('Y').mean()
    yearly_mean = temp_df.groupby('year')['Temp'].transform('mean')
    return yearly_mean.values

forType = 'daymet'
Ttrain = [19801001, 19951001] #training period
valid_date = [19951001, 20101001]  # Testing period
#define inputs
if forType == 'daymet':
  varF = ['dayl', 'prcp', 'srad', 'tmax', 'tmin', 'tmean', 'vp']
else:
  varF = ['dayl', 'prcp', 'srad', 'tmax', 'vp']

# Define attributes list
attrLst = [ 'p_mean','pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
            'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
            'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
            'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
            'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
            'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']

locs = pd.read_csv("gages_list.csv")
camels_bd = np.loadtxt('camels_bd.txt', delimiter=',')
def transfer2SWAT(train_x, train_c, Ttrain, locs, camels_bd):
    iida = np.array(julian_date(Ttrain[0], Ttrain[1]))
    iida = np.repeat(iida.reshape(1,-1), train_x.shape[0], axis=0)
    rhd = calculate_rh(train_x[:,:, 5], train_x[:,:,6])
    srad = train_x[:,:,2]*train_x[:,:,0]/1000000
    laimax = np.repeat(train_c[:,13].reshape(-1,1), train_x.shape[1], axis=1)
    laidiff = np.repeat(train_c[:,14].reshape(-1,1), train_x.shape[1], axis=1)
    laiday = calculate_laiday(laimax, laidiff, iida)
    tmp_an = np.zeros_like(train_x[:,:,5])

    swat_c = np.zeros((train_c.shape[0], 45))
    swat_c_dict = createDictFromKeys(swat.defaultKeys['attributes'], mtd='ref', dat=swat_c)
    swat_c_dict["sol_nly"][:] = 5 #sol_nly
    swat_c_dict["sol_silt"][:] = train_c[:,26] #silt_frac
    swat_c_dict["sol_mwc"][:] = train_c[:,24]*1000 #awc (mm)
    swat_c_dict["sol_clay"][:] = train_c[:,27] #sol_clay
    swat_c_dict["sol_bd"][:] = camels_bd[:,0] #sol_bd
    swat_c_dict["sol_k"][:] = train_c[:, 23]*10 #sol_k
    swat_c_dict["slsoil"][:] = 50 #slsoil
    swat_c_dict["dep_imp"][:] = train_c[:,20]*1000 #dep_imp
    swat_c_dict["sub_elev"][:] = train_c[:,9] #sub_elev
    swat_c_dict["lat"][:] = locs['LAT'] #lat

    swat_c2 = np.zeros((train_c.shape[0], int(swat_c[:,4][0]), 7))

    for i in range(train_x.shape[0]):
        tmp_an[i,:] = an_temp(train_x[i,:,5], Ttrain)
        swat_c2[i,:,0] = np.arange(train_c[i,21], 0, -train_c[i,21] / swat_c[i,4])[::-1]
        swat_c2[i,:,2] = train_c[i, 26]  # silt_frac
        swat_c2[i,:,5] = train_c[i, 24] * 1000  # awc (mm)
        swat_c2[i,:,1] = train_c[i, 27]  # sol_clay
        swat_c2[i,:,3] = camels_bd[i, 0]  # sol_bd (Mg/m3)
        swat_c2[i,:,4] = train_c[i, 23] * 10  # sol_k

    swat_x = np.stack((train_x[:,:,1], train_x[:,:,3], train_x[:,:,4], tmp_an, train_x[:,:,5], srad, laiday, iida, rhd), axis=2)

    return swat_x, swat_c, swat_c2

swat = SWAT()

params0 = ('cnn', 'sol_alb', 'surlag', 'tconc', 'sub_sftmp', 'rad_factor', 'sub_sftmp',
                                    'sub_smfmx', 'sub_smfmn', 'sub_smtmp', 'sub_timp', 'snocovmx', 'snocov1',
                                    'snocov2', 'ffc', 'alpha_bf', 'gw_delaye', 'gw_revap', 'gw_spyld', 'gwqmn',
                                     'rchrg_dp','revapmn', 'gwqmn', 'alpha_bfe_d', 'harg_petco', 'esco')
npar = len(params0)
nbasin, nt, nx = train_x.shape; nhru, nc = train_c.shape
p = createTensor((nhru, npar))+0.5 # in reality this could come from an NN
pDict = createDictFromKeys(params0, mtd='ref', dat=p) # all keys refer to the data in p

Ttrain = [19801001, 19951001] #training period
valid_date = [19951001, 20101001]  # Testing period
xx,cc,cc2 = transfer2SWAT(train_x, train_c, Ttrain, locs, camels_bd)
xx, cc, cc2 = [torch.tensor(tensor, dtype=dtype).to(device) for tensor in [xx, cc, cc2]]

#cc = torch.rand((nhru,46))
swat(xx, (cc,cc2), pDict)
#        out = self.HBV(x, parameters=hbvpara, staind=self.staind, tdlst=self.tdlst, mu=self.nmul, muwts=wts, rtwts=routpara,
#                          bufftime=self.inittime, routOpt=self.routOpt, comprout=self.comprout, dydrop=self.dydrop)
#iterLoop
#   x, c, p= data_loader()
#   pDict = createDictFromKeys
#   swat(train_x, train_c, pDict)
