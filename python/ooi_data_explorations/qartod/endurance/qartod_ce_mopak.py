#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the MOPAK data from the uncabled, Coastal Endurance Surface
    Moorings and Profilers and process the data to generate QARTOD Gross Range
    and Climatology test limits
"""
import dateutil.parser as parser
import numpy as np
import os
import pandas as pd
import pytz
import sys
import xarray as xr

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

from ooi_data_explorations.common import list_deployments, get_deployment_dates, get_annotations, get_vocabulary, \
    m2m_request, m2m_collect, add_annotation_qc_flags, N_CORES
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.qartod.qc_processing import process_gross_range, process_climatology, inputs, \
    ANNO_HEADER, CLM_HEADER, GR_HEADER


def process_date_range(dates, site=None, node=None, sensor=None, method=None, stream=None):
    start_date = dates[0].strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_date = dates[1].strftime('%Y-%m-%dT%H:%M:%S.000Z')
    r = m2m_request(site, node, sensor, method, stream, start_date, end_date)
    if r:
        data = m2m_collect(r, '.*MOPAK.*\\.nc$')
        if data:
            data = data.isel(time=slice(None, None, 100))
        return data
    else:
        return None


def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from the different data delivery methods for the
    3D accelerometer (MOPAK, and combines them, where appropriate, into a
    single, merged xarray data sets.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged: the merged and resampled (if appropriate) MOPAK dataset
    """
    # set the function for downloading the data
    print('##### Downloading the telemetered MOPAK data for %s #####' % site)
    part_telem = partial(process_date_range, site=site, node=node, sensor=sensor,
                         method='telemetered', stream='mopak_o_dcl_accel')
    telem = []
    deployments = list_deployments(site, node, sensor)
    for deploy in deployments:
        # get the start and end dates for the deployment and use them to determine the data to download and
        # process. split the deployment dates into 30 day chunks to avoid overly large data requests
        start, stop = get_deployment_dates(site, node, sensor, deploy)
        dates = pd.date_range(start, stop, freq='30D')
        dates = [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]
        with ThreadPoolExecutor(max_workers=N_CORES) as executor:
            frames = list(tqdm(executor.map(part_telem, dates), total=len(dates),
                               desc='Downloading and Processing the Data Files', file=sys.stdout))

        frames = [i for i in frames if i]
        if frames:
            telem.append(xr.concat(frames, 'time'))

    # combine the telemetered datasets
    if telem:
        telem = xr.concat(telem, 'time')
    else:
        telem = None

    print('##### Downloading the recovered_host MOPAK data for %s #####' % site)
    part_rhost = partial(process_date_range, site=site, node=node, sensor=sensor,
                         method='recovered_host', stream='mopak_o_dcl_accel_recovered')
    rhost = []
    deployments = list_deployments(site, node, sensor)
    for deploy in deployments:
        # get the start and end dates for the deployment and use them to determine the data to download and
        # process. split the deployment dates into 30 day chunks to avoid overly large data requests
        start, stop = get_deployment_dates(site, node, sensor, deploy)
        dates = pd.date_range(start, stop, freq='30D')
        dates = [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]
        with ThreadPoolExecutor(max_workers=N_CORES) as executor:
            frames = list(tqdm(executor.map(part_rhost, dates), total=len(dates), disable=True,
                               desc='Downloading and Processing the Data Files', file=sys.stdout))

        frames = [i for i in frames if i]
        if frames:
            rhost.append(xr.concat(frames, 'time'))

    # combine the recovered_hosts datasets
    if rhost:
        rhost = xr.concat(rhost, 'time')
    else:
        rhost = None

    # combine the datasets into a single, merged dataset
    merged = combine_datasets(telem, rhost, None, None)
    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load all MOPAK data for a defined reference designator (using the site,
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
    # load the combined data for the different sources of MOPAK data
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

    _, index = np.unique(data['time'], return_index=True)
    data = data.isel(time=index)
    data = data.sel(time=slice('2014-01-01T00:00:00', end_date))

    # set the parameters and the gross range limits
    parameters = ['mopak_accelx', 'mopak_accely', 'mopak_accelz',
                  'mopak_ang_ratex', 'mopak_ang_ratey', 'mopak_ang_ratez',
                  'mopak_magx', 'mopak_magy', 'mopak_magz']

    limits = [[-5, 5], [-5, 5], [-5, 5],
              [-5.24, 5.24], [-5.24, 5.24], [-5.24, 5.24],
              [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]]

    # create the initial gross range entry
    gr_lookup = process_gross_range(data, parameters, limits, site=site,
                                    node=node, sensor=sensor, stream='dosta_o_dcl_accel')

    # add the stream name and the source comment
    gr_lookup['notes'] = ('User range based on data collected through {}.'.format(src_date))

    # just use the first 3 parameters for the climatology test
    clm_lookup, clm_table = process_climatology(data, parameters[:3], limits[:3], depth_bins=None,
                                                site=site, node=node, sensor=sensor,
                                                stream='dosta_abcdjm_replace_me')

    # add the stream name
    clm_lookup['stream'] = 'dosta_abcdjm_replace_me'

    return annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the MOPAK data from the OOI M2M system and create the
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
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/qartod/mopak')
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
    parameters = ['mopak_accelx', 'mopak_accely', 'mopak_accelz']
    for i in range(len(parameters)):
        tbl = '-'.join([site, node, sensor, parameters[i]]) + '.csv'
        with open(os.path.join(out_path, tbl), 'w') as clm:
            clm.write(clm_table[i])


if __name__ == '__main__':
    main()
