#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the CTDBP data from the uncabled, Coastal Endurance Surface
    Moorings and processes the data to generate QARTOD Gross Range and
    Climatology test limits
"""
import dateutil.parser as parser
import os
import pandas as pd
import pytz

from ooi_data_explorations.common import get_annotations, load_gc_thredds, add_annotation_qc_flags
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.uncabled.process_ctdbp import ctdbp_datalogger, ctdbp_instrument
from ooi_data_explorations.qartod.qc_processing import process_gross_range, process_climatology, inputs, \
    ANNO_HEADER, CLM_HEADER, GR_HEADER


def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from each of the three data delivery methods for
    the uncabled CTD (CTDBP), and combines each of them into a single, merged
    xarray data set.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged: the merged CTDBP data stream resampled to a 3-hour time
        record
    """
    # download the telemetered data and re-process it to create a more useful and coherent data set
    tag = '.*CTDBP.*\\.nc$'
    telem = load_gc_thredds(site, node, sensor, 'telemetered', 'ctdbp_cdef_dcl_instrument', tag)
    telem = ctdbp_datalogger(telem)

    # download the recovered host data and re-process it to create a more useful and coherent data set
    rhost = load_gc_thredds(site, node, sensor, 'recovered_host', 'ctdbp_cdef_dcl_instrument_recovered', tag)
    rhost = ctdbp_datalogger(rhost)

    # download the recovered instrument data and re-process it to create a more useful and coherent data set
    rinst = load_gc_thredds(site, node, sensor, 'recovered_inst', 'ctdbp_cdef_instrument_recovered', tag)
    rinst = ctdbp_instrument(rinst)

    # combine the three datasets into a single, merged time series resampled to an hourly time series
    merged = combine_datasets(telem, rhost, rinst, 60)

    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load the CTD data for a defined reference designator (using the site, node
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

    # clean-up the data, removing values that were marked as fail either from the quality checks or in the
    # annotations, and all data collected after the cut off date
    data = data.where(data.rollup_annotations_qc_results != 4)

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

    # set the parameters and the sensor range limits
    parameters = ['sea_water_electrical_conductivity', 'sea_water_temperature',
                  'sea_water_pressure', 'sea_water_practical_salinity']

    if site == 'CE09OSSM' and node == 'MFD37':
        plimit = [0, 600]   # 600 m stain gauge pressure sensor
    else:
        plimit = [0, 100]   # 100 m stain gauge pressure sensor

    limits = [[0, 9], [-5, 35], plimit, [0, 42]]

    # create the initial gross range entry (lookup) table for the data using expanded gross range limits
    gr_lookup = process_gross_range(data, parameters, limits, expanded=True, site=site, node=node, sensor=sensor)

    # replicate it three times for the different streams
    gr_lookup = pd.concat([gr_lookup] * 3, ignore_index=True)

    # re-work the gross range entries for the different streams, resetting the variable names back to OOINet names
    streams = ['ctdbp_cdef_dcl_instrument', 'ctdbp_cdef_dcl_instrument_recovered', 'ctdbp_cdef_instrument_recovered']
    variables = [
        ['conductivity', 'temp', 'pressure', 'practical_salinity'],
        ['conductivity', 'temp', 'pressure', 'practical_salinity'],
        ['ctdbp_seawater_conductivity', 'ctdbp_seawater_temperature', 'ctdbp_seawater_pressure', 'practical_salinity']
    ]
    idx = 0
    for num, stream in enumerate(streams):
        for j in range(4):
            gr_lookup['parameter'][idx + j] = {'inp': variables[num][j]}
            gr_lookup['stream'][idx + j] = stream
        idx += 4

    # set the default source string
    gr_lookup['source'] = ('User Gross Range based on data collected from {} through to {}.'.format(start_date,
                                                                                                    src_date))

    # create the initial climatology lookup and tables for the data
    clm_lookup, clm_table = process_climatology(data, parameters[1:4:2], limits[1:4:2], expanded=True,
                                                site=site, node=node, sensor=sensor)

    # replicate the climatology lookup table three times for the different streams
    clm_lookup = pd.concat([clm_lookup] * 3, ignore_index=True)

    # re-work the climatology lookup table for the different streams, resetting the variable names back to OOINet names
    idx = 0
    for num, stream in enumerate(streams):
        for j in [1, 3]:
            clm_lookup['parameters'][idx] = {'inp': variables[num][j], 'tinp': 'time', 'zinp': 'None'}
            clm_lookup['stream'][idx] = stream
            idx += 1

    return annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the CTDBP data from the Gold Copy THREDDS server and create the
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
    parameters = ['sea_water_temperature', 'practical_salinity']
    for i in range(len(parameters)):
        tbl = '-'.join([site, node, sensor, parameters[i]]) + '.csv'
        with open(os.path.join(out_path, tbl), 'w') as clm:
            clm.write(clm_table[i])


if __name__ == '__main__':
    main()
