#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the CTDPF data from the uncabled, Coastal Endurance Profiler
    Mooring (and the CSPPs) and process the data to generate QARTOD Gross
    Range and Climatology test limits
"""
import dateutil.parser as parser
import os
import pandas as pd
import pytz
import sys
import xarray as xr

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from ooi_data_explorations.common import get_annotations, get_vocabulary, load_gc_thredds, add_annotation_qc_flags
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.profilers import split_profiles, bin_profiles
from ooi_data_explorations.uncabled.process_ctdpf import ctdpf_cspp, ctdpf_wfp
from ooi_data_explorations.qartod.qc_processing import process_gross_range, process_climatology, woa_standard_bins, \
    inputs, ANNO_HEADER, CLM_HEADER, GR_HEADER


def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from the different data delivery methods for the
    CTD and combines them into a single, merged xarray data set.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged: the merged CTDPF dataset
    """
    # set the tag to use for downloading the data
    tag = '.*CTDPF.*\\.nc$'

    if node == 'WFP01':
        # this CTDPF is part of a WFP and includes telemetered and recovered data
        n_cores = min(10, int(os.cpu_count() / 2) - 1)  # number of physical cores to use for the parallel processing
        binning = partial(bin_profiles, site_depth=520.0, bin_size=2.0)  # function object for the parallel processing

        print('##### Downloading the telemetered CTDPF data for %s #####' % site)
        telem = load_gc_thredds(site, node, sensor, 'telemetered', 'ctdpf_ckl_wfp_instrument', tag)
        drop_vars = ['qc_executed', 'qc_results', 'qartod_executed']
        for var in telem.variables:  # drop select qc variables
            if any([drop in var for drop in drop_vars]):
                telem = telem.drop_vars(var)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(telem.groupby('deployment'))
        for grp in grps:
            print('# -- Processing telemetered deployment %s' % grp[0])
            # preprocess the data to get it into a more useful format with numbered profiles
            data = ctdpf_wfp(grp[1])
            # split the data into a list of individual profiles
            profiles = split_profiles(data)
            profiles = [i for i in profiles if i]  # remove any empty profiles
            # bin the data in each profile into 2 m bins (using dask to parallelize the binning)
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                binned = list(tqdm(executor.map(binning, profiles), total=len(profiles),
                                   desc='Binning the Profiles', file=sys.stdout))
            # combine the binned profiles into a single dataset
            binned = [i for i in binned if i]
            binned = xr.concat(binned, 'time')
            binned = binned.sortby(['profile', 'time'])
            deployments.append(binned)
            del data, profiles, binned

        deployments = [i for i in deployments if i]
        telem = xr.concat(deployments, 'time')
        del deployments, grps

        print('##### Downloading the recovered_wfp CTDPF data for %s #####' % site)
        rhost = load_gc_thredds(site, node, sensor, 'recovered_wfp', 'ctdpf_ckl_wfp_instrument_recovered', tag)
        drop_vars = ['qc_executed', 'qc_results', 'qartod_executed']
        for var in rhost.variables:  # drop select qc variables
            if any([drop in var for drop in drop_vars]):
                rhost = rhost.drop_vars(var)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rhost.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_host deployment %s' % grp[0])
            # preprocess the data to get it into a more useful format with numbered profiles
            data = ctdpf_cspp(grp[1])
            # split the data into a list of individual profiles
            profiles = split_profiles(data)
            # bin the data in each profile into 2 m bins (using dask to parallelize the binning)
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                binned = list(tqdm(executor.map(binning, profiles), total=len(profiles),
                                   desc='Binning the Profiles', file=sys.stdout))
            # combine the binned profiles into a single dataset
            binned = [i for i in binned if i]
            binned = xr.concat(binned, 'time')
            binned = binned.sortby(['profile', 'time'])
            deployments.append(binned)
            del data, profiles, binned

        deployments = [i for i in deployments if i]
        rhost = xr.concat(deployments, 'time')
        del deployments, grps

        # merge, but do not resample the time records.
        merged = combine_datasets(telem, rhost, None, None)
        del telem, rhost
    elif node == 'SP001':
        # this CTDPF is part of a CSPP
        print('##### Downloading the recovered_cspp CTDPF data for %s #####' % site)
        rinst = load_gc_thredds(site, node, sensor, 'recovered_cspp', 'ctdpf_j_cspp_instrument_recovered', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rinst.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_cspp deployment %s' % grp[0])
            # preprocess the data to get it into a more useful format with numbered profiles
            deployments.append(ctdpf_cspp(grp[1]))

        deployments = [i for i in deployments if i]
        merged = xr.concat(deployments, 'time')
    else:
        return ValueError('Unrecognized node designator')

    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load all CTDPF data for a defined reference designator (using the site,
    node and sensor names to construct the reference designator) and
    collected via the different data delivery methods and combine them into a
    single data set from which QARTOD test limits for the gross range and
    climatology tests can be calculated.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :param cut_off: string formatted date to use as cut-off for data to add
        to QARTOD test sets
    :return gr_lookup: CSV formatted strings to save to a csv file for the
        QARTOD gross range lookup tables.
    :return clm_lookup: CSV formatted strings to save to a csv file for the
        QARTOD climatology lookup tables.
    :return clm_table: CSV formatted strings to save to a csv file for the
        QARTOD climatology range tables.
    """
    # load the combined data for the different sources of CTDPF data
    data = combine_delivery_methods(site, node, sensor)

    # get the current system annotations for the sensor
    annotations = get_annotations(site, node, sensor)
    annotations = pd.DataFrame(annotations)
    if not annotations.empty:
        annotations = annotations.drop(columns=['@class'])
        annotations['beginDate'] = pd.to_datetime(annotations.beginDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
        annotations['endDate'] = pd.to_datetime(annotations.endDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

        # create an annotation-based quality flag
        data = add_annotation_qc_flags(data, annotations)

    if 'rollup_annotations_qc_results' in data.variables:
        data = data.where(data.rollup_annotations_qc_results != 4, drop=True)

    # if a cut_off date was used, limit data to all data collected up to the cut_off date.
    # otherwise, set the limit to the range of the downloaded data.
    if cut_off:
        cut = parser.parse(cut_off)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
        src_date = cut.strftime('%Y-%m-%d')
    else:
        cut = parser.parse(data.time_coverage_end)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
        src_date = cut.strftime('%Y-%m-%d')

    data = data.sel(time=slice('2014-01-01T00:00:00', end_date))

    # set the parameters and the gross range limits
    parameters = ['sea_water_electrical_conductivity', 'sea_water_temperature',
                  'sea_water_pressure', 'sea_water_practical_salinity']
    if node == 'WFP01':
        plimits = [0, 600]  # sensor limits on the MMP
    else:
        plimits = [0, 350]  # sensor limits on the CSPP
    limits = [[0, 9], [-5, 35], plimits, [0, 42]]

    # create the initial gross range entry
    gr_lookup = process_gross_range(data, parameters, limits, site=site, node=node, sensor=sensor,
                                    stream='ctdpf_INSERT_STREAM_NAME', extended=True)

    # add the stream name and the source comment
    gr_lookup['notes'] = ('User range based on data collected through {}.'.format(src_date))

    # set up the bins for a depth based climatology
    vocab = get_vocabulary(site, node, sensor)[0]
    max_depth = vocab['maxdepth']
    depth_bins = woa_standard_bins()
    m = depth_bins[:, 1] <= max_depth
    depth_bins = depth_bins[m, :]

    # set the parameters and the climatology limits
    parameters = ['sea_water_temperature', 'sea_water_practical_salinity']
    limits = [[-5, 35], [0, 42]]

    # create and format the climatology lookups and tables for the data
    clm_lookup, clm_table = process_climatology(data, parameters, limits, depth_bins=depth_bins,
                                                site=site, node=node, sensor=sensor,
                                                stream='ctdpf_INSERT_STREAM_NAME', extended=True)

    return annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the CTDPF data from the Gold Copy THREDDS server and create the
    QARTOD gross range and climatology test lookup tables.
    """
    # set up the input arguments
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    cut_off = args.cut_off

    # create the QARTOD gross range and climatology lookup values and tables
    annotations, gr_lookup, clm_lookup, clm_table = generate_qartod(site, node, sensor, cut_off)

    # save the downloaded annotations and qartod lookups and tables
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/qartod/ctdpf')
    out_path = os.path.abspath(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # save the annotations to a csv file for further processing
    anno_csv = '-'.join([site, node, sensor]) + '.quality_annotations.csv'
    annotations.to_csv(os.path.join(out_path, anno_csv), index=False, columns=ANNO_HEADER)

    # save the gross range values to a csv for further processing
    gr_csv = '-'.join([site, node, sensor]) + '.gross_range.csv'
    gr_lookup.to_csv(os.path.join(out_path, gr_csv), index=False, columns=GR_HEADER, float_format='%g')

    # save the climatology values and table to a csv for further processing
    clm_csv = '-'.join([site, node, sensor]) + '.climatology.csv'
    clm_lookup.to_csv(os.path.join(out_path, clm_csv), index=False, columns=CLM_HEADER)
    parameters = ['sea_water_temperature', 'sea_water_practical_salinity']
    for i in range(len(parameters)):
        tbl = '-'.join([site, node, sensor, parameters[i]]) + '.csv'
        with open(os.path.join(out_path, tbl), 'w') as clm:
            clm.write(clm_table[i])


if __name__ == '__main__':
    main()
