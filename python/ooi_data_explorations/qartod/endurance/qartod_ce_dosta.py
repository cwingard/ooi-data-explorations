#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the DOSTA data from the uncabled, Coastal Endurance Surface
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
from ooi_data_explorations.uncabled.process_dosta import dosta_datalogger, dosta_ctdbp_datalogger, \
    dosta_ctdbp_instrument, dosta_cspp
from ooi_data_explorations.qartod.qc_processing import process_gross_range, \
    process_climatology, woa_standard_bins, inputs, ANNO_HEADER, CLM_HEADER, GR_HEADER


def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from the different data delivery methods for the
    dissolved oxygen sensors (DOSTA, DOFST), and combines them, where appropriate,
    into a single, merged xarray data sets.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged: the merged and resampled (if appropriate) DOSTA dataset
    """
    # set the stream and tag constants
    tag = '.*DOSTA.*\\.nc$'

    if node == 'SP001':
        # this DOSTA is part of a CSPP and includes recovered data only
        print('##### Downloading the recovered_cspp DOSTA data for %s #####' % site)
        rhost = load_gc_thredds(site, node, sensor, 'recovered_cspp', 'dosta_abcdjm_cspp_instrument_recovered', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rhost.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_host deployment %s' % grp[0])
            deployments.append(dosta_cspp(grp[1]))
        deployments = [i for i in deployments if i]
        rhost = xr.concat(deployments, 'time')

        # merge, but do not resample the time records.
        merged = combine_datasets(None, rhost, None, None)
    elif node in ['RID16', 'MFD37']:
        # this DOSTA is connected to a CTDBP and includes all 3 types of data delivery methods
        print('##### Downloading the telemetered DOSTA data for %s #####' % site)
        telem = load_gc_thredds(site, node, sensor, 'telemetered', 'dosta_abcdjm_ctdbp_dcl_instrument', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(telem.groupby('deployment'))
        for grp in grps:
            print('# -- Processing telemetered deployment %s' % grp[0])
            deployments.append(dosta_ctdbp_datalogger(grp[1]))
        deployments = [i for i in deployments if i]
        telem = xr.concat(deployments, 'time')

        print('##### Downloading the recovered_host DOSTA data for %s #####' % site)
        rhost = load_gc_thredds(site, node, sensor, 'recovered_host', 'dosta_abcdjm_ctdbp_dcl_instrument_recovered', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rhost.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_host deployment %s' % grp[0])
            deployments.append(dosta_ctdbp_datalogger(grp[1]))
        deployments = [i for i in deployments if i]
        rhost = xr.concat(deployments, 'time')

        print('##### Downloading the recovered_inst DOSTA data for %s #####' % site)
        rinst = load_gc_thredds(site, node, sensor, 'recovered_inst', 'dosta_abcdjm_ctdbp_instrument_recovered', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rinst.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_inst deployment %s' % grp[0])
            deployments.append(dosta_ctdbp_instrument(grp[1]))
        deployments = [i for i in deployments if i]
        rinst = xr.concat(deployments, 'time')

        # merge and resample to a 2 hour data record
        merged = combine_datasets(telem, rhost, rinst, 120)
    else:
        # this DOSTA is standalone on one of the NSIFs and includes the telemetered and recovered_host data.
        # the data is collected in bursts (3 minutes at 1 Hz). process each data set per-deployment
        print('##### Downloading the telemetered DOSTA data for %s #####' % site)
        telem = load_gc_thredds(site, node, sensor, 'telemetered', 'dosta_abcdjm_dcl_instrument', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(telem.groupby('deployment'))
        for grp in grps:
            print('# -- Processing telemetered deployment %s' % grp[0])
            deployments.append(dosta_datalogger(grp[1], True))
        deployments = [i for i in deployments if i]
        telem = xr.concat(deployments, 'time')

        print('##### Downloading the recovered_host DOSTA data for %s #####' % site)
        rhost = load_gc_thredds(site, node, sensor, 'recovered_host', 'dosta_abcdjm_dcl_instrument_recovered', tag)
        deployments = []
        print('# -- Group the data by deployment and process the data')
        grps = list(rhost.groupby('deployment'))
        for grp in grps:
            print('# -- Processing recovered_host deployment %s' % grp[0])
            deployments.append(dosta_datalogger(grp[1], True))
        deployments = [i for i in deployments if i]
        rhost = xr.concat(deployments, 'time')

        # combine the datasets, leaving them as 15-minute median averaged datasets
        merged = combine_datasets(telem, rhost, None, None)

    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load all DOSTA data for a defined reference designator (using the site,
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
    # load the combined data for the different sources of DOSTA data
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

    # clean-up the data, NaN-ing values that were marked as fail in the QC checks and then removing all
    # records where the rollup annotation (every parameter fails) was set to fail.
    if 'oxygen_concentration_corrected_qc_summary_flag' in data.variables:
        m = data.oxygen_concentration_corrected_qc_summary_flag == 4
        data['oxygen_concentration'][m] = np.nan
        data['oxygen_concentration_corrected'][m] = np.nan
        if 'svu_oxygen_concentration' in data.variables:
            data['svu_oxygen_concentration'][m] = np.nan

    if 'dissolved_oxygen_annotations_qc_results' in data.variables:
        m = data.dissolved_oxygen_annotations_qc_results == 4
        data['oxygen_concentration'][m] = np.nan
        data['oxygen_concentration_corrected'][m] = np.nan
        if 'svu_oxygen_concentration' in data.variables:
            data['svu_oxygen_concentration'][m] = np.nan

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
    parameters = ['oxygen_concentration', 'svu_oxygen_concentration', 'oxygen_concentration_corrected']
    limits = [[0, 500], [0, 500], [0, 500]]

    # create the initial gross range entry
    gr_lookup = process_gross_range(data, parameters, limits, stdx=5, site=site,
                                    node=node, sensor=sensor, stream='dosta_abcdjm_replace_me',
                                    fixed_lower=True)

    # add the stream name and the source comment
    gr_lookup['source'] = ('User range based on data collected through {}.'.format(src_date))

    # based on the site and node, determine if we need a depth based climatology
    depth_bins = np.array([])
    if node == 'SP001':
        vocab = get_vocabulary(site, node, sensor)[0]
        max_depth = vocab['maxdepth']
        depth_bins = woa_standard_bins()
        m = depth_bins[:, 1] <= max_depth
        depth_bins = depth_bins[m, :]

    # create and format the climatology lookups and tables for the data
    clm_lookup, clm_table = process_climatology(data, parameters, limits, stdx=5, depth_bins=depth_bins,
                                                site=site, node=node, sensor=sensor,
                                                stream='dosta_abcdjm_replace_me',
                                                fixed_lower=True)

    # add the stream name
    clm_lookup['stream'] = 'dosta_abcdjm_replace_me'

    return annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the DOSTA data from the Gold Copy THREDDS server and create the
    QARTOD gross range and climatology test lookup tables.
    """
    # setup the input arguments
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    cut_off = args.cut_off

    # create the QARTOD gross range and climatology lookup values and tables
    annotations, gr_lookup, clm_lookup, clm_table = generate_qartod(site, node, sensor, cut_off)

    # save the downloaded annotations and qartod lookups and tables
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/qartod/dosta')
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
    if node in ['RID16', 'MFD37']:
        parameters = ['oxygen_concentration', 'oxygen_concentration_corrected']
    else:
        parameters = ['oxygen_concentration', 'svu_oxygen_concentration', 'oxygen_concentration_corrected']
    for i in range(len(parameters)):
        tbl = '-'.join([site, node, sensor, parameters[i]]) + '.csv'
        with open(os.path.join(out_path, tbl), 'w') as clm:
            clm.write(clm_table[i])


if __name__ == '__main__':
    main()
