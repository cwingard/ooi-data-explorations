#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the FDCHP data from the uncabled, Coastal Endurance Surface
    Moorings and processes the data to generate QARTOD Gross Range and
    Climatology test limits
"""
import dateutil.parser as parser
import os
import pandas as pd
import pytz

from ooi_data_explorations.common import get_annotations, load_gc_thredds, add_annotation_qc_flags
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.qartod.qc_processing import process_gross_range, process_climatology, inputs, \
    ANNO_HEADER, CLM_HEADER, GR_HEADER


def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from each of the two data delivery methods for
    the direct, covariance flux sensor (FDCHP), and combines each of
    them into a single, merged xarray data set.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged: the FDCHP data from the telemetered and recovered_host
    """
    # download the telemetered and recovered_host data and combine them into a single data set
    tag = '.*FDCHP.*\\.nc$'
    telem = load_gc_thredds(site, node, sensor, 'telemetered', 'fdchp_a_dcl_instrument', tag)
    rhost = load_gc_thredds(site, node, sensor, 'recovered_host', 'fdchp_a_dcl_instrument_recovered', tag)

    # combine the two datasets into a single, merged time series
    merged = combine_datasets(telem, rhost, None, None)
    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load all FDCHP data for a defined reference designator (using the site,
    node and sensor names to construct the reference designator) collected
    from the telemetered and recovered_host methods and combine them into a
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
    :return ssw_table: CSV formatted strings to save to a csv file for the
        QARTOD climatology range tables.
    """
    # load the combined telemetered and recovered_host data
    data = combine_delivery_methods(site, node, sensor)

    # get the current system annotations for the sensor
    annotations = get_annotations(site, node, sensor)
    annotations = pd.DataFrame(annotations)
    if not annotations.empty:
        annotations = annotations.drop(columns=['@class'])
        annotations['beginDate'] = pd.to_datetime(annotations.beginDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
        annotations['endDate'] = pd.to_datetime(annotations.endDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

    # create an annotation-based quality flag for the data
    data = add_annotation_qc_flags(data, annotations)

    # remove all records where the entire data set was marked as fail
    data = data.where(data.rollup_annotations_qc_results != 4)

    # remove filled values from the data set
    data = data.where(data.wind_speed > -99)

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

    # create the initial gross range entries.
    parameters = ['speed_of_sound_avg', 'wind_speed', 'wind_u_avg', 'wind_v_avg', 'wind_w_avg', 'vw_momentum_flux',
                  'uw_momentum_flux', 'buoyance_flux']
    limits = [[300, 370], [0, 65], [-65, 65], [-65, 65], [-65, 65], [-10, 10], [-20, 10], [-2.3, 2.3]]

    gr_lookup = process_gross_range(data, parameters, limits, site=site, node=node, sensor=sensor)

    # replicate it twice for the different streams
    gr_lookup = pd.concat([gr_lookup] * 2, ignore_index=True)

    # re-work the gross range entries for the different streams
    streams = ['fdchp_a_dcl_instrument', 'fdchp_a_dcl_instrument_recovered']
    idx = 0
    for num, stream in enumerate(streams):
        for j in range(len(parameters)):
            gr_lookup['stream'][idx + j] = stream
        idx += len(parameters)

    # add the source details
    gr_lookup['source'] = ('User range based on data collected through {}.'.format(src_date))

    # create the initial climatology entries
    clm_lookup, clm_table = process_climatology(data, parameters, limits, site=site, node=node, sensor=sensor)

    # replicate it twice for the different streams
    clm_lookup = pd.concat([clm_lookup] * 2, ignore_index=True)

    # re-work the climatology entries for the different streams
    idx = 0
    for num, stream in enumerate(streams):
        for j in range(len(parameters)):
            clm_lookup['stream'][idx + j] = stream
        idx += len(parameters)

    # add the source details
    clm_lookup['source'] = ('Climatology test values based on data collected through {}.'.format(src_date))

    return annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the FDCHP data from the Gold Copy THREDDS server and create the
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
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/qartod/fdchp')
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
    parameters = ['speed_of_sound_avg', 'wind_speed', 'wind_u_avg', 'wind_v_avg', 'wind_w_avg', 'vw_momentum_flux',
                  'uw_momentum_flux', 'buoyance_flux']
    for i in range(len(parameters)):
        tbl = '-'.join([site, node, sensor, parameters[i]]) + '.csv'
        with open(os.path.join(out_path, tbl), 'w') as clm:
            clm.write(clm_table[i])


if __name__ == '__main__':
    main()
