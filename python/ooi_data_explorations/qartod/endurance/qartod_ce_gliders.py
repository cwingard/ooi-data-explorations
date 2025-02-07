#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load Coastal Endurance glider data from the IOOS GilderDAC for use in
    creating QARTOD test values for different sensors either on the gliders
    or for co-located systems which may be seasonally sparse in data coverage,
    like the CSPP.
"""
import numpy as np
import pandas as pd
import requests
import sys
import warnings
import xarray as xr

from concurrent.futures import ThreadPoolExecutor
from erddapy import ERDDAP
from functools import partial
from tqdm import tqdm

# set up an ERDDAP server object for the IOOS GliderDAC
GLIDER_DAC = ERDDAP(server='ngdac')

# TODO: Add methods for the ADCPA, CTDGV, DOSTA, FLORT, and PARAD data sets

def create_box(lat, lon, extent=2.7):
    """

    :param lat:
    :param lon:
    :param extent:
    :return:
    """
    bounding_box = [
        lat - (extent / 60.),
        lat + (extent / 60.),
        lon - (extent * np.cos(np.radians(lat)) / 60.),
        lon + (extent * np.cos(np.radians(lat)) / 60.)
    ]

    return bounding_box


def list_gliders(bounding_box):
    """
    """
    advanced_search = {
        'institution': 'ooi_coastal_endurance',
        'min_lat': bounding_box[0],
        'max_lat': bounding_box[1],
        'min_lon': bounding_box[2],
        'max_lon': bounding_box[3],
    }

    search_url = GLIDER_DAC.get_search_url(response='csv', search_for='delayed', **advanced_search)
    search = pd.read_csv(search_url)
    gliders = search["Dataset ID"].values
    return gliders


def download_glider(dataset_id, bounding_box):
    """
    Download the glider data from the GliderDAC, optionally using a latitude
    and longitude box to further constrain the request.

    :param bounding_box:
    :param dataset_id:
    :return:
    """
    # set up the components of the ERDDAP request
    GLIDER_DAC.constraints = {
        'latitude>=': bounding_box[0],
        'latitude<=': bounding_box[1],
        'longitude>=': bounding_box[2],
        'longitude<=': bounding_box[3],
    }
    GLIDER_DAC.protocol = 'tabledap'
    GLIDER_DAC.variables = ['precise_time', 'precise_lon', 'precise_lat', 'depth', 'pressure', 'temperature',
                            'conductivity', 'salinity', 'density', 'backscatter', 'CDOM', 'chlorophyll',
                            'dissolved_oxygen', 'PAR']
    GLIDER_DAC.dataset_id = dataset_id

    # load the data into an xarray dataset
    try:
        ds = GLIDER_DAC.to_xarray()
    except requests.exceptions.HTTPError:
        # message = f"No data found within the bounding box for dataset ID {dataset_id}."
        # warnings.warn(message)
        return None

    ds = ds.swap_dims({'obs': 'precise_time'})
    ds = ds.reset_coords()
    keys = ['profile_id', 'time', 'longitude', 'latitude', 'trajectoryIndex', 'rowSize']
    for key in keys:
        if key in ds.variables:
            ds = ds.drop_vars(key)

    # rename some parameters to align with the other data sets (PARAD and FLORT)
    ds = ds.squeeze(drop=True)
    ds = ds.rename({
        'precise_time': 'time',
        'precise_lon': 'longitude',
        'precise_lat': 'latitude',
        'PAR': 'par',
        'backscatter': 'bback',
        'chlorophyll': 'estimated_chlorophyll',
        'CDOM': 'fluorometric_cdom'
    })

    # sort the data by time
    ds = ds.sortby('time')

    return ds


def collect_glider(latitude, longitude):
    """

    :param latitude:
    :param longitude:
    :return:
    """
    # based on the bounding box, create a list of glider datasets to download
    bounding_box = create_box(latitude, longitude)
    gliders = list_gliders(bounding_box)

    # download the data for each of the datasets
    partial_glider = partial(download_glider, bounding_box=bounding_box)
    with ThreadPoolExecutor(max_workers=5) as executor:
        frames = list(tqdm(executor.map(partial_glider, gliders), total=len(gliders),
                           desc='Downloading and Processing the Glider Data', file=sys.stdout))

    data = xr.concat([i for i in frames if i], dim='time')
    data = data.sortby('time')
    return data
