



# import datetime as dt
# import glob
import os
# import pprint
# import sys
# import matplotlib as mpl
# import matplotlib.colors
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# import seaborn as sns
import xarray as xr

# # import bnn_tools.bnn_array
#
# plt.style.use('default')
# xr.set_options(
#     display_expand_data=False,
#     display_expand_data_vars=True,
#     display_max_rows=10,
#     display_style='html',
#     display_width=80,
#     display_expand_attrs=False
# );
# # endregion

# %%
import nais_netcdf
import nais_netcdf.bnn_array
import nais_netcdf.constants as co


# %%
FLAG_ID = co.FLAG_ID

DP = co.DP

DNDLDP = co.DNDLDP

TIME = co.TIME

FLAGS = co.FLAGS

MODE = co.MODE

POL_DIC = co.POL_DIC

MODE_DIC = co.MODE_DIC
P_ION = co.P_ION

MODES = co.MODES
LDP = co.LDP
META = co.META

def get_mode_ds(data_path, day, log_dy, mode, sec_resample):
    pos_neg = POL_DIC[mode]
    ion_par = MODE_DIC[mode]
    name = f'NAIS{pos_neg}{day}{ion_par}.sum'
    dndldp_df, fl_df = get_dndldp_flag_dfs(data_path, name)

    flags_ds = flag_df2ds(fl_df, sec_resample)

    dndldp_ds = dndldp_df2ds(dndldp_df, log_dy, sec_resample)

    ds = merge_dndldp_fl(dndldp_ds, flags_ds, mode)
    return ds




def merge_dndldp_fl(dndldp_ds, flags_ds, mode):
    ds = (
        xr.merge([dndldp_ds, flags_ds])
        .expand_dims({MODE: [mode]})
        .bnn.set_Dp()
    )
    return ds


def dndldp_df2ds(dndldp, log_dy, sec_resample):
    data = (
        dndldp
        .stack()
        ._set_name(name=DNDLDP)
        .to_xarray()
        # .pipe(lambda d:
        #     d.where(d>0,.0001).where(d.notnull()))
        .bnn.resample_ts(sec_resample)
        .bnn.dp_regrid(n_subs=10, log_dy=log_dy)
        .bnn.set_Dp()
        .drop(LDP)
        # .bnn.plot_psd()
        # [DP]
    )
    return data


def flag_df2ds(fl_df, sec_resample):
    dt_secs = sec_resample
    flags = (
        fl_df.to_frame()
        .reset_index()
        .assign(**{
            'sec':
                lambda d:
                (
                    pd.Series(
                        (d[TIME]
                            # + pd.Timedelta(dt_secs / 2, 'seconds')
                            ))
                    .dt.round(pd.Timedelta(dt_secs, 'seconds'))
                )
        })
        [[FLAGS, 'sec']]
        .set_index('sec')
        .groupby('sec')
        .agg(sum)
        .rename_axis(index=TIME)
        .to_xarray()
    )
    return flags


def get_dndldp_flag_dfs(data_path, name):
    d = pd.read_csv(os.path.join(data_path, name), index_col=0)
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d.columns.name = DP
    fl = d[FLAGS]
    dndldp = d.drop(FLAGS, axis=1)
    dndldp.columns = pd.Series(dndldp.columns).astype(float)
    return dndldp, fl

def get_full_ds(data_path, day, log_dy, sec_resample):
    flag_ds = get_flag_id_ds(data_path, day)

    dss = []
    for mode in MODES:
        dndldp_df = get_mode_ds(
            data_path, day, log_dy, mode, sec_resample)
        dss.append(dndldp_df)

    ds_comb:xr.Dataset = (
        xr.concat(dss, dim=MODE)
        .pipe(lambda d: xr.merge([d,flag_ds]))
    )

    for k,v in META.items():
        ds_comb[k] = ds_comb[k].assign_attrs(v)




    return ds_comb

def get_flag_id_ds(data_path, day):
    flag_path = os.path.join(data_path, f'NAIS{day}.flags')
    df = (
        pd.read_csv(flag_path, index_col=0)
        .rename_axis(FLAG_ID)
    )
    return df.to_xarray()

def save_ds2nc(ds_comb, data_path_out, day):
    (
        ds_comb
        .to_netcdf(os.path.join(data_path_out, f'NAIS{day}.nc'))
    )



# %%
def main():
    # %%
    pass
    # %%
    data_path_in = os.path.join(nais_netcdf.__path__[0], 'data_in')
    data_path_out = os.path.join(nais_netcdf.__path__[0], 'data_out')

    # %%
    log_dy = .025
    sec_resample = 300
    day = '20220720'

    ds_comb = get_full_ds(data_path_in, day, log_dy, sec_resample)

    save_ds2nc(ds_comb, data_path_out, day)







