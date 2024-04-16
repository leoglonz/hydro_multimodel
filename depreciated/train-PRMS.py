"""
-------------------------------------
Code pulled and reorganized from `PGML_STemp > HydroModels`
for demonstration purposes.

Credit Fang et al.
-------------------------------------

Purpose: This file runs training on the PRMS PyTorch model for the 15-year testing
window of 1995/10/01 to 2010/09/30 (or [19951001,20101001]) with all 671 CAMELS 
basins by default.

Default settings:


"""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from post import plot, stat
from sklearn import preprocessing
