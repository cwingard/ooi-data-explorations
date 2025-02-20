#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the WAVSS data from the uncabled, Coastal Endurance Surface
    Moorings and processes the data to generate QARTOD Gross Range and
    Climatology test limits
"""
import dateutil.parser as parser
import os

import numpy as np
import pandas as pd
import pytz
import xarray as xr

from ooi_data_explorations.common import get_annotations, load_gc_thredds, add_annotation_qc_flags
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.uncabled.process_wavss import wavss_datalogger
from ooi_data_explorations.qartod.qc_processing import process_gross_range, process_climatology, inputs, \
    ANNO_HEADER, CLM_HEADER, GR_HEADER


def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from each of the two data delivery methods for
    the bulk wave statistics from the Tri-Axys wave sensor (or just the
    recovered instrument data for the tRDI WavesMon data), and combines each
    of them into a single, merged xarray data set.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged: the combined, merged telemetered and recovered_host data
    """
    if node == 'SBD12':
        # this is a Tri-Axys wave sensor, so set the tag and stream names
        tag = '.*WAVSS.*\\.nc$'
        tstream = 'wavss_a_dcl_statistics'
        rstream = 'wavss_a_dcl_statistics_recovered'

        # download the telemetered data and re-process it to create a more useful and coherent data set
        telem = load_gc_thredds(site, node, sensor, 'telemetered', tstream, tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(telem.groupby('deployment'))
        for grp in grps:
            print('# -- Processing telemetered deployment %s' % grp[0])
            deployments.append(wavss_datalogger(grp[1]))
        deployments = [i for i in deployments if i]
        telem = xr.concat(deployments, 'time')

        # download the recovered host data and re-process it to create a more useful and coherent data set
        rhost = load_gc_thredds(site, node, sensor, 'recovered_host', rstream, tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rhost.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_host deployment %s' % grp[0])
            deployments.append(wavss_datalogger(grp[1]))
        deployments = [i for i in deployments if i]
        rhost = xr.concat(deployments, 'time')

        # combine the two datasets into a single, merged time series (no resampling)
        merged = combine_datasets(telem, rhost, None, None)
    elif node == 'MFD35':
        # this is the tRDI ADCP, so set the tag and stream names and download the recovered instrument data
        merged = load_gc_thredds(site, node, sensor, 'recovered_inst', 'adcpt_m_instrument_log9_recovered', '.*\\.nc$')
    else:
        return ValueError('Node not recognized.')

    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load all the wave data for a defined reference designator (using the
    site, node and sensor names to construct the reference designator)
    collected via the recovered instrument method and combine them into a
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
        QARTOD climatology range tables for each parameter.
    """
    # load the combined telemetered and recovered_host data
    wavss = combine_delivery_methods(site, node, sensor)

    # the CF card on CE02SHSM was corrupted during deployment 4, all of that deployment's data is bad
    if site == 'CE02SHSM':
        wavss = wavss.where(wavss.deployment != 4, drop=True)

    # get the current system annotations for the sensor
    annotations = get_annotations(site, node, sensor)
    annotations = pd.DataFrame(annotations)
    if not annotations.empty:
        annotations = annotations.drop(columns=['@class'])
        annotations['beginDate'] = pd.to_datetime(annotations.beginDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
        annotations['endDate'] = pd.to_datetime(annotations.endDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

    # create an annotation-based quality flag
    wavss = add_annotation_qc_flags(wavss, annotations)

    # clean-up the wavss data, removing values that marked as fail in the annotations
    wavss = wavss.where(wavss.rollup_annotations_qc_results != 4)

    # if a cut_off date was used, limit data to all data collected up to the cut_off date.
    # otherwise, set the limit to the range of the downloaded data.
    if cut_off:
        cut = parser.parse(cut_off)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
        src_date = cut.strftime('%Y-%m-%d')
    else:
        cut = parser.parse(wavss.time_coverage_end)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
        src_date = cut.strftime('%Y-%m-%d')

    wavss = wavss.sel(time=slice('2014-01-01T00:00:00', end_date))

    # set up the parameters and fail limits for the gross range and climatology tests
    if node == 'SBD12':
        parameters = ['significant_wave_height', 'average_wave_height', 'max_wave_height', 'wave_height_10',
                      'wave_height_hmo', 'mean_spectral_period', 'mean_wave_period', 'peak_wave_period',
                      'significant_period', 'wave_period_10', 'wave_period_tp5', 'mean_direction', 'mean_spread']
        limits = [[0, 40], [0, 40], [0, 40], [0, 40], [0, 40], [1.5, 33], [1.5, 33],
                  [1.5, 33], [1.5, 33], [1.5, 33], [1.5, 33], [0, 360], [0, 90]]

        # convert the directional values to radians prior to processing
        wavss['mean_direction'] = np.deg2rad(wavss['mean_direction'])
    else:
        parameters = ['significant_wave_height', 'peak_wave_period', 'peak_wave_direction', 'h_mean', 't_mean',
                      'd_mean', 'h_max', 't_max', 'h_1_3', 't_1_3', 'h_1_10', 't_1_10', 'hs_sea', 'tp_sea',
                      'dp_sea', 'hs_swell', 'tp_swell', 'dp_swell']
        limits = [[0, 40], [1.5, 33], [0, 360], [0, 40], [1.5, 33], [0, 360], [0, 40], [1.5, 33], [0, 40],
                  [1.5, 33], [0, 40], [1.5, 33], [0, 40], [1.5, 33], [0, 360], [0, 40], [1.5, 33], [0, 360]]

        # convert the directional values to radians prior to processing
        wavss['peak_wave_direction'] = np.deg2rad(wavss['peak_wave_direction'])
        wavss['d_mean'] = np.deg2rad(wavss['d_mean'])
        wavss['dp_sea'] = np.deg2rad(wavss['dp_sea'])
        wavss['dp_swell'] = np.deg2rad(wavss['dp_swell'])

    # create and format the gross range lookups for the wavss streams
    gr_lookup = process_gross_range(wavss, parameters, limits, site=site, node=node, sensor=sensor, extended=True)
    gr_lookup['source'] = ('User range based on data collected through {}.'.format(src_date))

    # create and format the climatology lookups and tables for the wavss streams
    clm_lookup, clm_table = process_climatology(wavss, parameters, limits, site=site, node=node, sensor=sensor,
                                                extended=True)
    clm_lookup['source'] = ('Monthly ranges based on data collected through {}.'.format(src_date))

    return annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the WAVSS data from the Gold Copy THREDDS server and create the
    QARTOD gross range and climatology test lookup tables.
    """
    # set up the input arguments
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    cut_off = args.cut_off

    # create the initial HITL annotation blocks, the QARTOD gross range and climatology lookup values, and
    # the climatology table for the pco2_seawater parameter
    annotations, gr_lookup, clm_lookup, clm_table = generate_qartod(site, node, sensor, cut_off)

    # save the resulting annotations and qartod lookups and tables
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/qartod/wavss')
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
    if node == 'SBD12':
        parameters = ['significant_wave_height', 'average_wave_height', 'max_wave_height', 'wave_height_10',
                      'wave_height_hmo', 'mean_spectral_period', 'mean_wave_period', 'peak_wave_period',
                      'significant_period', 'wave_period_10', 'wave_period_tp5', 'mean_direction', 'mean_spread']
    else:
        parameters = ['significant_wave_height', 'peak_wave_period', 'peak_wave_direction', 'h_mean', 't_mean',
                      'd_mean', 'h_max', 't_max', 'h_1_3', 't_1_3', 'h_1_10', 't_1_10', 'hs_sea', 'ts_sea',
                      'dp_sea', 'hs_swell', 'ts_swell', 'dp_swell']
    for i in range(len(parameters)):
        tbl = '-'.join([site, node, sensor, parameters[i]]) + '.csv'
        with open(os.path.join(out_path, tbl), 'w') as clm:
            clm.write(clm_table[i])


if __name__ == '__main__':
    main()
