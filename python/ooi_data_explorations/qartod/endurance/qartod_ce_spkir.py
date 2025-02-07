#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the SPKIR data from the uncabled, Coastal Endurance Surface
    Moorings and Profilers and process the data to generate QARTOD Gross Range
    and Climatology test limits
"""
import dateutil.parser as parser
import numpy as np
import os
import pandas as pd
import pytz
import xarray as xr

from ooi_data_explorations.common import get_annotations, get_vocabulary, load_gc_thredds, add_annotation_qc_flags
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.uncabled.process_spkir import spkir_datalogger, spkir_cspp
from ooi_data_explorations.qartod.qc_processing import process_gross_range, process_climatology, \
    woa_standard_bins, inputs, ANNO_HEADER, CLM_HEADER, GR_HEADER

# TODO: Add a method to pull in the profile data and run the qartod tests on it

def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from the different data delivery methods for the
    seven-channel, downwelling spectral irradiance (SPKIR) sensor, and combines
    them into a single, merged xarray data sets.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged: the merged and resampled (if appropriate) SPKIR dataset
    """
    # set the stream and tag constants
    tag = '.*SPKIR.*\\.nc$'
    if node == 'SP001':
        print('##### Downloading the recovered_cspp SPKIR data for %s #####' % site)
        rhost = load_gc_thredds(site, node, sensor, 'recovered_cspp', 'spkir_abj_cspp_instrument_recovered', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rhost.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_cspp deployment %s' % grp[0])
            deployments.append(spkir_cspp(grp[1]))
        deployments = [i for i in deployments if i]
        merged = xr.concat(deployments, 'time')
        merged = merged.sortby(['deployment', 'time'])
        _, index = np.unique(merged['time'], return_index=True)
        merged = merged.isel(time=index)
    else:
        # this SPKIR is standalone on one of the NSIFs and includes telemetered and recovered_host data.
        # data is collected in bursts (3 minutes at 1 Hz). process each data set per-deployment
        print('##### Downloading the telemetered SPKIR data for %s #####' % site)
        telem = load_gc_thredds(site, node, sensor, 'telemetered', 'spkir_abj_dcl_instrument', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(telem.groupby('deployment'))
        for grp in grps:
            print('# -- Processing telemetered deployment %s' % grp[0])
            deployments.append(spkir_datalogger(grp[1], True))
        deployments = [i for i in deployments if i]
        telem = xr.concat(deployments, 'time')

        print('##### Downloading the recovered_host SPKIR data for %s #####' % site)
        rhost = load_gc_thredds(site, node, sensor, 'recovered_host', 'spkir_abj_dcl_instrument_recovered', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rhost.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_host deployment %s' % grp[0])
            deployments.append(spkir_datalogger(grp[1], True))
        deployments = [i for i in deployments if i]
        rhost = xr.concat(deployments, 'time')

        # combine the datasets, leaving them as 15-minute median averaged datasets
        merged = combine_datasets(telem, rhost, None, None)

    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load all SPKIR data for a defined reference designator (using the site,
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
    # load the combined data for the different sources of SPKIR data
    data = combine_delivery_methods(site, node, sensor)

    # get the current system annotations for the sensor
    annotations = get_annotations(site, node, sensor)
    annotations = pd.DataFrame(annotations)
    if not annotations.empty:
        annotations = annotations.drop(columns=['@class'])
        annotations['beginDate'] = pd.to_datetime(annotations.beginDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
        annotations['endDate'] = pd.to_datetime(annotations.endDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

    # create annotation-based quality flags and remove any data that has been flagged as bad (removing the flag
    # variables after they are used)
    data = add_annotation_qc_flags(data, annotations)
    if 'rollup_annotations_qc_results' in data.variables:
        data = data.where(data.rollup_annotations_qc_results != 4)
        data = data.drop('rollup_annotations_qc_results')

    if 'spkir_abj_cspp_downwelling_vector_annotations_qc_results' in data.variables:
        data = data.where(data.spkir_abj_cspp_downwelling_vector_annotations_qc_results != 4)
        data = data.drop('spkir_abj_cspp_downwelling_vector_annotations_qc_results')

    if 'channel_array_annotations_qc_results' in data.variables:
        data = data.where(data.channel_array_annotations_qc_results != 4)
        data = data.drop('channel_array_annotations_qc_results')

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
    start_date = str(data.time[0].values.min())[:10]

    # NaN out downwelling irradiance data measured outside solar noon. Will only use the nominal max values from
    # each day (approximately solar noon) to create the gross range and climatology (min value will always be 0).
    data = data.where((data.time.dt.hour > 18) & (data.time.dt.hour < 22))

    # set the parameters and the gross range limits
    parameters = ['downwelling_irradiance_412', 'downwelling_irradiance_444', 'downwelling_irradiance_490',
                  'downwelling_irradiance_510', 'downwelling_irradiance_555', 'downwelling_irradiance_620',
                  'downwelling_irradiance_683']
    limits = [[0, 300], [0, 300], [0, 300], [0, 300], [0, 300], [0, 300], [0, 300]]

    # create the initial gross range entry
    gr_lookup = process_gross_range(data, parameters, limits, site=site,
                                    node=node, sensor=sensor, stream='temporary_spkir_stream_name',
                                    fixed_lower=True)

    # add the source comment
    gr_lookup['source'] = ('User Gross Range based on data collected from {} through to {}.'.format(start_date,
                                                                                                    src_date))

    # based on the node, determine if we need a depth based climatology
    depth_bins = np.array([])
    if node == 'SP001':
        vocab = get_vocabulary(site, node, sensor)[0]
        max_depth = vocab['maxdepth']
        depth_bins = woa_standard_bins()
        m = depth_bins[:, 1] <= max_depth
        depth_bins = depth_bins[m, :]

    # create and format the climatology lookups and tables for the data
    clm_lookup, clm_table = process_climatology(data, parameters, limits, depth_bins=depth_bins,
                                                site=site, node=node, sensor=sensor,
                                                stream='temporary_spkir_stream_name',
                                                fixed_lower=True)

    # add the source comment
    clm_lookup['source'] = ('Climatology based on data collected from {} through to {}.'.format(start_date, src_date))

    return data, annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the SPKIR data from the Gold Copy THREDDS server and create the
    QARTOD gross range and climatology test lookup tables.
    """
    # set up the input arguments
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    cut_off = args.cut_off

    # create the QARTOD gross range and climatology lookup values and tables
    data, annotations, gr_lookup, clm_lookup, clm_table = generate_qartod(site, node, sensor, cut_off)

    # save the downloaded annotations and qartod lookups and tables
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/qartod/spkir')
    out_path = os.path.abspath(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # save the annotations to a csv file for further processing
    anno_csv = '-'.join([site, node, sensor]) + '.quality_annotations.csv'
    annotations.to_csv(os.path.join(out_path, anno_csv), index=False, columns=ANNO_HEADER)

    # save the gross range values to a csv for further processing
    gr_csv = '-'.join([site, node, sensor]) + '.gross_range.csv'
    gr_lookup.to_csv(os.path.join(out_path, gr_csv), index=False, columns=GR_HEADER)

    # save the climatology values and table to a csv for further processing
    clm_csv = '-'.join([site, node, sensor]) + '.climatology.csv'
    clm_lookup.to_csv(os.path.join(out_path, clm_csv), index=False, columns=CLM_HEADER)
    parameters = ['downwelling_irradiance_412', 'downwelling_irradiance_444', 'downwelling_irradiance_490',
                  'downwelling_irradiance_510', 'downwelling_irradiance_555', 'downwelling_irradiance_620',
                  'downwelling_irradiance_683']
    for i in range(len(parameters)):
        tbl = '-'.join([site, node, sensor, parameters[i]]) + '.csv'
        with open(os.path.join(out_path, tbl), 'w') as clm:
            clm.write(clm_table[i])


if __name__ == '__main__':
    main()
