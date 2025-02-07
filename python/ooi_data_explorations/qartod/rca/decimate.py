# -*- coding: utf-8 -*-
"""decimate.py

This module contains code for performing LTTB decimation.

"""
import gc
import math
import numba
import numpy as np
import pandas as pd
from typing import Callable, Optional
import logging
import xarray as xr
import dask.array as darray

from functools import reduce


@numba.njit
def split_data(data: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Splits the data by the number of bins

    Parameters
    ----------
    data : np.ndarray
        The data to be split
    n_bins : int
        The number of bins the data should be divided into

    Returns
    -------
    np.ndarray
        The split data arrays
    """
    return np.array_split(data[1 : len(data) - 1], n_bins)


@numba.njit
def _np_apply_along_axis(
    func1d: Callable, axis: int, arr: np.ndarray
) -> np.ndarray:
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@numba.njit
def np_mean(array: np.ndarray, axis: int) -> np.ndarray:
    """Simple numpy mean re-creation for numba"""
    return _np_apply_along_axis(np.mean, axis, array)


@numba.njit
def np_median(array: np.ndarray, axis: int) -> np.ndarray:
    """Simple numpy median re-creation for numba"""
    return _np_apply_along_axis(np.median, axis, array)


@numba.njit
def _areas_of_triangles(a: np.ndarray, bs: np.ndarray, c: np.ndarray):
    """Calculate areas of triangles from duples of vertex coordinates.

    Uses implicit numpy broadcasting along first axis of ``bs``.

    Parameters
    ----------
    a : np.ndarray
        Point a
    bs : np.ndarray
        Current data bin
    c : np.ndarray
        The mean of the next bin
    Returns
    -------
    numpy.array
        Array of areas of shape (len(bs),)
    """
    bs_minus_a = bs - a
    a_minus_bs = a - bs
    return 0.5 * np.absolute(
        (a[0] - c[0]) * (bs_minus_a[:, 1]) - (a_minus_bs[:, 0]) * (c[1] - a[1])
    )


@numba.njit
def _largest_triangle_three_buckets(
    data: np.ndarray, threshold: int
) -> np.array:
    """
    Return a downsampled version of data.
    Original code found at https://github.com/devoxi/lttb-py.

    Parameters
    ----------
    data : np.ndarray
        Original data that will be decimated.
        Must be a numpy array or list of lists.
        Data must be formatted this way: [[x,y], [x,y], [x,y], ...]
    threshold : int
        threshold must be >= 2 and <= to the len of data.
    Returns
    -------
    numpy.ndarray
        Decimated data.
    """
    n_bins = threshold - 2

    # Prepare output array
    # First and last points are the same as in the input.
    out = np.zeros((threshold, 2))
    out[0] = data[0]
    out[len(out) - 1] = data[len(data) - 1]

    data_bins = split_data(data, n_bins)

    # Largest Triangle Three Buckets (LTTB):
    # In each bin, find the point that makes the largest triangle
    # with the point saved in the previous bin
    # and the centroid of the points in the next bin.
    for i in range(len(data_bins)):
        this_bin = data_bins[i]
        if i < n_bins - 1:
            next_bin = data_bins[i + 1]
        else:
            next_bin = data[len(data) - 1 :]

        a = out[i]
        bs = this_bin
        c = np_mean(next_bin, 0)
        areas = _areas_of_triangles(a, bs, c)

        # Get middle of bucket
        middle = bs[math.floor(bs.shape[0] / 2)]
        point = bs[np.argmax(areas)]

        out[i + 1] = np.array([middle[0], point[1]])
    return out


class LttbException(Exception):
    pass


def _perform_decimation(ds, threshold):
    time_da = ds.time.astype(int)
    cols = [time_da.name, ds.name]
    data = darray.stack([time_da.data, ds.data], axis=1).compute()
    try:
        decdata = _largest_triangle_three_buckets(data, threshold)
    except Exception as e:
        raise LttbException(e)
    del ds
    gc.collect()
    return pd.DataFrame(decdata, columns=cols)


def downsample(
    raw_ds: xr.Dataset,
    threshold: int,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Performs downsampling on the dataset.

    Parameters
    ----------
    raw_ds : xr.Dataset
        The dataset to be decimated.
    threshold : int
        The threshold for decimation,
        total number of data points at the end.
    logger : logging.Logger
        Logger instance to be used for logging

    Returns
    -------
    pd.DataFrame
        The decimated data as pandas dataframe.
    """
    if logger is None:
        from loguru import logger
    logger.info("Get list of data arrays")
    da_list = (raw_ds[var] for var in raw_ds)

    df_list = []
    for da in da_list:
        logger.info(f"Executing decimation for {da.name}")
        decdf = _perform_decimation(da, threshold)
        df_list.append(decdf)

        del decdf
        gc.collect()
    logger.info("Decimation process completed.")

    logger.info("Creating decimated dataframe.")
    final_df = reduce(
        lambda left, right: pd.merge(left, right, on="time"), df_list
    )
    final_df['time'] = pd.to_datetime(final_df['time'])

    return final_df
