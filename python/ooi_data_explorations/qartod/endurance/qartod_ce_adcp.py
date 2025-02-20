#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the ADCP data from the uncabled, Coastal Endurance Surface
    Moorings and processes the data to generate QARTOD Gross Range and
    Climatology test limits
"""
import dateutil.parser as parser
import numpy as np
import os
import pandas as pd
import pytz

from ooi_data_explorations.common import get_annotations, get_vocabulary, load_gc_thredds, add_annotation_qc_flags
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.qartod.qc_processing import process_gross_range, process_climatology, inputs, \
    woa_standard_bins, ANNO_HEADER, CLM_HEADER, GR_HEADER


def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from each of the three data delivery methods for
    the uncabled CTD (ADCP), and combines each of them into a single, merged
    xarray data set.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged: the merged ADCP data stream resampled to a 3-hour time
        record
    """
    # download the telemetered data and re-process it to create a more useful and coherent data set
    tag = '.*ADCP.*\\.nc$'
    stream = 'adcp_velocity_earth'
    telem = load_gc_thredds(site, node, sensor, 'telemetered', stream, tag)

    # download the recovered host data and re-process it to create a more useful and coherent data set
    rhost = load_gc_thredds(site, node, sensor, 'recovered_host', stream, tag)

    # download the recovered instrument data and re-process it to create a more useful and coherent data set
    rinst = load_gc_thredds(site, node, sensor, 'recovered_inst', stream, tag)

    # combine the three datasets into a single, merged time series
    merged = combine_datasets(telem, rhost, rinst, None)
    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load the ADCP data for a defined reference designator (using the site, node
    and sensor names to construct the reference designator) collected via the
    telemetered, recovered host and instrument methods and combine them into a
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
        QARTOD climatology range table for the seafloor pressure and temperature.
    """
    # load the combined telemetered and recovered_host data for the data and water streams
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

    # apply some basic sensor specific checks to the data
    data = data.where(np.abs(data['pitch']) < 1500, drop=True)
    data = data.where(np.abs(data['roll']) < 1500, drop=True)

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

    # set the parameters and the sensor range limits
    parameters = ['pitch','roll','temperature','eastward_seawater_velocity','northward_seawater_velocity']
    limits = [[-2000, 2000], [-2000, 2000], [-500, 4500], [-5, 5], [-5, 5]]

    # create the initial gross range entry
    gr_lookup = process_gross_range(data, parameters, limits, site=site, node=node, sensor=sensor,
                                    stream='adcpt_velocity_earth', extended=True)

    # add the source date to the notes
    gr_lookup['notes'] = ('User range based on data collected through {}.'.format(src_date))

    # set up the bins for a depth based climatology
    vocab = get_vocabulary(site, node, sensor)[0]
    max_depth = vocab['maxdepth']
    depth_bins = woa_standard_bins()
    m = depth_bins[:, 1] <= max_depth
    depth_bins = depth_bins[m, :]
    
    # create the initial climatology lookup and tables for the data
    # TODO: need to flatten the bin_depths and velocity data to allow for the climatology processing to work
    data = data.rename({'bin_depths': 'depth'})
    clm_lookup, clm_table = process_climatology(data, parameters[3:], limits[3:], site=site, node=node,
                                                depth_bins=depth_bins, sensor=sensor, stream='adcpt_velocity_earth')

    return annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the ADCP data from the Gold Copy THREDDS server and create the
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
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/qartod/ctdbp')
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
    parameters = ['eastward_seawater_velocity', 'northward_seawater_velocity']
    for i in range(len(parameters)):
        tbl = '-'.join([site, node, sensor, parameters[i]]) + '.csv'
        with open(os.path.join(out_path, tbl), 'w') as clm:
            clm.write(clm_table[i])


if __name__ == '__main__':
    main()
