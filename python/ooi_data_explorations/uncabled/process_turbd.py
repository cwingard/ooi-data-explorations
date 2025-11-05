#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@brief Functions to reprocess and clean up the TURBD datasets downloaded from OOINet
"""

import numpy as np
import pandas as pd
import xarray as xr

from ooi_data_explorations.common import FILL_INT, FILL_FLOAT, ENCODINGS

ATTRS = {
    'raw_backscatter': {
        'long_name': 'Raw Optical Backscatter at 700 nm',
        'units': 'counts',
        'comment': 'Raw optical backscatter measurements at 700 nm.',
        'data_product_identifier': 'FLUBSCT_L0',
        'coordinates': 'time lat lon'
    }
}


def turbd_datalogger(ds, burst=False):
    """
    Takes TURBD data recovred by the data loggers used in CGSN moorings
    and cleeans up the data set to make it more user friendly. Primary 
    task is renaming parameters and dropping some that are of limited use.

    This function:
    - Drops redundant variables
    - Renames variables for clarity
    - Updates variable attributes
    - Optionally resamples burst-mode data to median values, with associated statistics

    Parameters
    ----------
    ds : xarray.Dataset
        TURBD dataset with raw signal and calculated turbidity.
    burst : bool, default=False
        If True, resample burst-mode data to median over 15 min intervals and
        compute min, max, and std for turbidity.

    Returns
    -------
    ds: xarray.Dataset
        Cleaned up dataset
    """
    # Drop some redundant variables
    #   internal_timestamp == superceded by time
    #   measurement_wavelength_beta == constant, add as attribute to raw_signal_beta
    drop_vars = {'internal_timestamp', 'measurement_wavelength_beta'} & set(ds.variables)
    if drop_vars:
        ds = ds.drop_vars(drop_vars)

    # Rename variables in one call
    rename_map = {'raw_signal_beta': 'raw_backscatter'}
    rename_map = {k: v for k, v in rename_map.items() if k in ds.variables}
    if rename_map:
        ds = ds.rename(rename_map)
        # Keep original variable name as attribute
        for old_name, new_name in rename_map.items():
            ds[new_name].attrs['ooinet_variable_name'] = old_name

    # Update attributes only for variables that exist
    for var, attrs in ATTRS.items():
        if var in ds.variables:
            ds[var].attrs.update(attrs)

    if burst:
        # Shift time so that burst windows align, then resample
        ds = ds.assign_coords(time=ds['time'] + np.timedelta64(450, 's'))
        burst_ds = ds.resample(time='900s', skipna=True).median(keep_attrs=True)

        # Turbidity statistics in one go
        turbd_resampled = ds['turbidity'].resample(time='900s', skipna=True)
        turbd_stats = xr.concat(
            [turbd_resampled.min(), turbd_resampled.max(), turbd_resampled.std()],
            dim='stats'
        )
        turbd_stats = turbd_stats.assign_coords(stats=('stats', [0, 1, 2]))

        # Merge stats into burst dataset
        burst_ds['turbidity_burst_stats'] = turbd_stats
        burst_ds = burst_ds.where(~np.isnan(burst_ds.deployment), drop=True)

        # Add attributes
        burst_ds['turbidity_burst_stats'].attrs.update({
            'long_name': 'Turbidity Statistics',
            'units': 'ntu',
            'comment': 'Associated statistics for the resampled turbidity measurements.',
            'data_product_identifier': 'FLUBSCT_L0',
            'statistics': 'min max std'
        })

        ds = burst_ds

    return ds

