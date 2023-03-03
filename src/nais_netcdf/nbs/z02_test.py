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
import nais_netcdf.constants as co


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

META = co.META

# %%
# def main():
    # pass

# %%
# path to data in
data_path_in = os.path.join(nais_netcdf.__path__[0], 'data_in')
# path to data out
data_path_out = os.path.join(nais_netcdf.__path__[0], 'data_out')

# %%
log_dy = .025
sec_resample = 300
day = '20220720'

ds_comb = fu.get_full_ds(data_path_in, day, log_dy, sec_resample)

fu.save_ds2nc(ds_comb, data_path_out, day)


# %%
# this is what is inside get_full_ds

flag_ds = fu.get_flag_id_ds(data_path_in, day)

dss = []
for mode in MODES:
    dndldp_df = fu.get_mode_ds(
        data_path_in, day, log_dy, mode, sec_resample)
    dss.append(dndldp_df)

ds_comb:xr.Dataset = (
    xr.concat(dss, dim=MODE)
    .pipe(lambda d: xr.merge([d,flag_ds]))
)

for k,v in META.items():
    ds_comb[k] = ds_comb[k].assign_attrs(v)


# %%
# def get_mode_ds(data_path, day, log_dy, mode, sec_resample):
mode = P_ION
pos_neg = POL_DIC[mode]
ion_par = MODE_DIC[mode]
name = f'NAIS{pos_neg}{day}{ion_par}.sum'
dndldp_df, fl_df = fu.get_dndldp_flag_dfs(data_path_in, name)

flags_ds = fu.flag_df2ds(fl_df, sec_resample)

dndldp_ds = fu.dndldp_df2ds(dndldp_df, log_dy, sec_resample)

ds = fu.merge_dndldp_fl(dndldp_ds, flags_ds, mode)


# %%
ds

# %%
