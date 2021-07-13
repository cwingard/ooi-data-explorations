#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief combines data sets with different delivery methods (e.g. telemetered and
    recovered instrument) into a single integrated dataset.
"""
import argparse
import glob
import numpy as np
import os
import sys
import xarray as xr


def combine_datasets(tdata, rhdata, ridata, resample_time):
    """
    Load and merge data from telemetered, recovered host and recovered
    instrument data sets. Telemetered and recovered host data represent the
    same source of data, just different data delivery methods. These data files
    are concatenated together and only unique time records are kept. The
    recovered instrument data is concatenated onto the telemetered/recovered
    host data set and then the full data set is resampled to a common time
    record via median averaging. The resulting merged and resampled data set is
    returned for further analysis.

    :param tdata: telemetered data as an xarray data set or None if no data available
    :param rhdata: recovered host data as an xarray data set or None if no data available
    :param ridata: recovered instrument data as xarray data set or None if no data available
    :param resample_time: The resampling time period in minutes
    :return ds: The combined and resampled data set
    """
    # combine the telemetered and recovered host datasets, which have the same variables,
    # dropping any 1-dimensional variables along the way.
    if tdata and rhdata:
        # use concat to join the datasets and then select only unique time points
        ds = xr.concat([tdata.squeeze(), rhdata.squeeze()], 'time')
        _, index = np.unique(ds['time'], return_index=True)
        ds = ds.isel(time=index)
    elif tdata and not rhdata:
        # telemetered data, but no recovered host data
        ds = tdata.squeeze()
    elif rhdata and not tdata:
        # recovered host data, but no telemetered data
        ds = rhdata.squeeze()
    else:
        # no telemetered or recovered host data
        ds = None

    # combine the recovered instrument data with the telemetered/recovered host data, if both exists
    if ds and ridata:
        # drop any one-dimensional variables in ridata and make sure we have a unique time record
        ridata = ridata.squeeze()
        _, index = np.unique(ridata['time'], return_index=True)
        ridata = ridata.isel(time=index)

        # first, identify any variables in ds that are not available in ridata
        for v in ds.variables:
            if v not in ridata.variables:
                # add an empty variable of the same type and dimensions to ridata
                ridata[v] = ds[v].broadcast_like(ridata['time'])

        # next, identify any variables in ridata that are not available in ds
        for v in ridata.variables:
            if v not in ds.variables:
                # add an empty variable of the same type and dimensions to ridata
                ds[v] = ridata[v].broadcast_like(ds['time'])

        # finally, concat the datasets and remove any duplicate timestamps
        ds = xr.concat([ds, ridata], 'time')
        _, index = np.unique(ds['time'], return_index=True)
        ds = ds.isel(time=index)
    elif ds and not ridata:
        pass
    elif ridata and not ds:
        # no telemetered/recovered host data, just the recovered instrument data.
        ds = ridata
    else:
        return None

    # resample the dataset onto a common time record
    itime = '{:d}Min'.format(resample_time)
    btime = int(resample_time / 2)
    loff = '{:d}Min'.format(btime)
    gtime = '{:d}Min'.format(resample_time * 3)
    ds = ds.sortby('time')
    avg = ds.resample(time=itime, base=btime, loffset=loff, skipna=True).median(keep_attrs=True)
    avg = avg.interpolate_na(dim='time', max_gap=gtime)

    # add the attributes back into the data set
    avg.attrs = ds.attrs
    for v in avg.variables:
        if v != 'time':
            avg[v] = avg[v].astype(ds[v].dtype)
            avg[v].attrs = ds[v].attrs

    avg.time.attrs['long_name'] = 'Time'
    avg.time.attrs['standard_name'] = 'time'
    avg.time.attrs['axis'] = 'T'

    return avg


def inputs(argv=None):
    """
    Parses the command line arguments for combining datasets.

    :param argv: Command line input arguments
    :return args: Parsed command line arguments
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize argument parser
    parser = argparse.ArgumentParser(
        description="""Combine datasets from different data delivery methods""")

    # assign input arguments.
    parser.add_argument("-d", "--data_dir", dest="ddir", type=str, required=True)
    parser.add_argument("-t", "--telemetered", dest="telem", default=False, action='store_true')
    parser.add_argument("-rh", "--recovered_host", dest="rhost", default=False, action='store_true')
    parser.add_argument("-ri", "--recovered_inst", dest="rinst", default=False, action='store_true')
    parser.add_argument("-dp", "--deployment", dest="deploy", type=int, required=True)
    parser.add_argument("-rt", "--resample_time", dest="resample", type=int, required=True)
    parser.add_argument("-o", "--outfile", dest="outfile", type=str, required=True)

    # parse the input arguments and create a parser object
    args = parser.parse_args(argv)

    return args


def main(argv=None):
    """
    Load and merge data from telemetered, recovered host and recovered 
    instrument data files downloaded to the local machine on a per deployment
    basis. Telemetered and recovered host data represent the same source of 
    data, just different data delivery methods. These data files are 
    concatenated together and only unique time records are kept. The recovered
    instrument data is concatenated onto the telemetered/recovered host data
    record and then the full data set is resampled to a common time record via 
    median averaging. The resulting merged and resampled data set is saved to
    disk for further analysis.
    
    :param argv:
    :return None:
    """
    args = inputs(argv)
    data_directory = args.ddir
    telemetered = args.telem
    recovered_host = args.rhost
    recovered_inst = args.rinst
    deployment = args.deploy
    resample_time = args.resample
    outfile = args.outfile

    # load the data from the different data delivery methods
    tdata = None
    rhdata = None
    ridata = None
    if telemetered:
        # load the telemetered data file
        tfile = glob.glob(os.path.join(data_directory, '*.deploy{:02d}.telemetered.*.nc'.format(deployment)))
        if tfile:
            tdata = xr.load_dataset(tfile[0])

    if recovered_host:
        # load the recovered_host data file
        rhfile = glob.glob(os.path.join(data_directory, '*.deploy{:02d}.recovered_host.*.nc'.format(deployment)))
        if rhfile:
            rhdata = xr.load_dataset(rhfile[0])

    if recovered_inst:
        # load the recovered_inst data file
        rifile = glob.glob(os.path.join(data_directory, '*.deploy{:02d}.recovered_inst.*.nc'.format(deployment)))
        if rifile:
            ridata = xr.load_dataset(rifile[0])

    # combine the data into a single dataset and save the combined and resampled data to disk
    if tdata or rhdata or ridata:
        # combine the data sets
        ds = combine_datasets(tdata, rhdata, ridata, resample_time)

        # save the combined and resampled data to disk
        outfile = os.path.join(data_directory, outfile)
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))

        ds.to_netcdf(outfile, mode='w', format='NETCDF4', engine='h5netcdf')


if __name__ == '__main__':
    main()
