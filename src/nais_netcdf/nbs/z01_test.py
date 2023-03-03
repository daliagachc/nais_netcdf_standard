# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:nais_netcdf_standard]
#     language: python
#     name: conda-env-nais_netcdf_standard-py
# ---

# %%
# region imports
from IPython import get_ipython

import nais_netcdf.constants as co

# noinspection PyBroadException
try:
    _magic = get_ipython().run_line_magic
    _magic("load_ext", "autoreload")
    _magic("autoreload", "2")
except:
    pass

# import datetime as dt
# import glob
import os
# import pprint
# import sys
# import matplotlib as mpl
# import matplotlib.colors
import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
import xarray as xr

# import bnn_tools.bnn_array


# %%
import nais_netcdf
import nais_netcdf.bnn_array
import nais_netcdf.funs as fu


# %%

DP = co.DP

DNDLDP = co.DNDLDP

TIME = co.TIME

FLAGS = co.FLAGS

MODE = co.MODE

POL_DIC = co.POL_DIC

MODE_DIC = co.MODE_DIC
P_ION = co.P_ION

MODES = co.MODES

# %%
# def main():
    # pass

# %%
data_path_in = os.path.join(nais_netcdf.__path__[0], 'data_in')
data_path_out = os.path.join(nais_netcdf.__path__[0], 'data_out')

# %%
log_dy = .025
sec_resample = 300
day = '20220720'

ds_comb = fu.get_full_ds(data_path_in, day, log_dy, sec_resample)

fu.save_ds2nc(ds_comb, data_path_out, day)


# %%
(
    ds_comb
    .loc[{MODE:P_ION}]
    [DNDLDP]
    .bnn.plot_psd()
)



# %%
(
    ds_comb
    .loc[{MODE:co.P_PAR}]
    [DNDLDP]
    .bnn.plot_psd()
)


# %%
