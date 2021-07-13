#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the PHSEN data from the uncabled, Coastal Endurance Surface
    Moorings and processes the data to generate QARTOD Gross Range and
    Climatology test limits
"""
import os
import pandas as pd

from ooi_data_explorations.common import get_annotations, load_gc_thredds, add_annotation_qc_flags
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.uncabled.process_phsen import phsen_datalogger, phsen_instrument, quality_checks
from ooi_data_explorations.qartod.qc_processing import identify_blocks, create_annotations, process_gross_range, \
    process_climatology, inputs


def combine_delivery_methods(site, node, sensor):
    """
    Takes the downloaded data from each of the three data delivery methods and
    combines them into a single, merged xarray data set.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :return merged:
    """
    # download the telemetered data and re-process it to create a more useful and coherent data set
    tag = '.*PHSEN.*\\.nc$'
    telem = load_gc_thredds(site, node, sensor, 'telemetered', 'phsen_abcdef_dcl_instrument', tag)
    telem = phsen_datalogger(telem)

    # download the recovered host data and re-process it to create a more useful and coherent data set
    rhost = load_gc_thredds(site, node, sensor, 'recovered_host', 'phsen_abcdef_dcl_instrument_recovered', tag)
    rhost = phsen_datalogger(rhost)

    # download the recovered instrument data and re-process it to create a more useful and coherent data set
    rinst = load_gc_thredds(site, node, sensor, 'recovered_inst', 'phsen_abcdef_instrument', tag)
    rinst = phsen_instrument(rinst)

    # combine the three datasets into a single, merged time series resampled to a 3 hour interval time series
    merged = combine_datasets(telem, rhost, rinst, 180)

    # re-run the quality checks, since averaging will change the flag values
    merged['seawater_ph_quality_flag'] = quality_checks(merged)
    return merged


def generate_qartod(site, node, sensor, cut_off):
    """
    Load all of the pH data for a defined reference designator (using the site,
    node and sensor names to construct the reference designator) collected via
    the three data delivery methods of telemetered, recovered host and
    recovered instrument and combine them into a single data set from which
    QARTOD test limits for the gross range and climatology tests can be
    calculated.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :param cut_off: string formatted date to use as cut-off for data to add
        to QARTOD test sets
    :return annotations: Initial list of auto-generated HITL annotations as
        a pandas dataframe
    :return gr_lookup: CSV formatted strings to save to a csv file for the
        QARTOD gross range lookup tables.
    :return clm_lookup: CSV formatted strings to save to a csv file for the
        QARTOD climatology lookup tables.
    :return clm_table: CSV formatted strings to save to a csv file for the
        QARTOD climatology range tables.
    """
    # load and combine all of the data sources for the pH sensor
    data = combine_delivery_methods(site, node, sensor)

    # create a boolean array of the data marked as "fail" by the pH quality checks and generate initial
    # HITL annotations that can be combined with system annotations and pH quality checks to create
    # a cleaned up data set prior to calculating the QARTOD test values
    fail = data.seawater_ph_quality_flag.where(data.seawater_ph_quality_flag == 4).notnull()
    blocks = identify_blocks(fail, [24, 24])
    hitl = create_annotations(site, node, sensor, blocks)

    # get the current system annotations for the sensor
    annotations = get_annotations(site, sensor, node)
    annotations = pd.DataFrame(annotations)
    if not annotations.empty:
        annotations = annotations.drop(columns=['@class'])
        annotations['beginDate'] = pd.to_datetime(annotations.beginDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
        annotations['endDate'] = pd.to_datetime(annotations.endDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

    # append the fail annotations to the existing annotations
    annotations = annotations.append(pd.DataFrame(hitl), ignore_index=True, sort=False)

    # create a roll-up annotation flag
    data = add_annotation_qc_flags(data, annotations)

    # clean-up the data, removing values that fail the pH quality checks or were marked as fail in the annotations
    data = data.where((data.seawater_ph_quality_flag != 4) & (data.rollup_annotations_qc_results != 4))

    # limit data to all data collected upto a preset cut_off date
    data = data.sel(time=slice("2014-01-01T00:00:00", cut_off))

    # create the initial gross range entry
    gr = process_gross_range(data, ['seawater_ph'], [6.9, 9.0], site=site, node=node, sensor=sensor)

    # re-work gross entry for the different streams and parameter names
    gr_lookup = pd.DataFrame()
    gr_lookup = gr_lookup.append([gr, gr, gr], ignore_index=True)
    gr_lookup['parameter'][0] = {'inp': 'phsen_abcdef_ph_seawater'}
    gr_lookup['stream'][0] = 'phsen_abcdef_dcl_instrument'
    gr_lookup['parameter'][1] = {'inp': 'phsen_abcdef_ph_seawater'}
    gr_lookup['stream'][1] = 'phsen_abcdef_dcl_instrument_recovered'
    gr_lookup['parameter'][2] = {'inp': 'phsen_abcdef_ph_seawater'}
    gr_lookup['stream'][2] = 'phsen_abcdef_instrument'
    gr_lookup['source'] = ('Sensor min/max pulled from the vendor processing code. The user min/max is the historical '
                           'mean of all data collected through NOMINAL_DATE +/- 3 standard deviations.')

    # create and format the climatology entry and table
    cll, clm_table = process_climatology(data, ['seawater_ph'], [6.9, 9.0], site=site, node=node, sensor=sensor)

    # re-work climatology entry for the different streams and parameter names
    clm_lookup = pd.DataFrame()
    clm_lookup = clm_lookup.append([cll, cll, cll])
    clm_lookup['parameters'][0] = {'inp': 'phsen_abcdef_ph_seawater', 'tinp': 'time', 'zinp': 'None'}
    clm_lookup['stream'][0] = 'phsen_abcdef_dcl_instrument'
    clm_lookup['parameters'][1] = {'inp': 'phsen_abcdef_ph_seawater', 'tinp': 'time', 'zinp': 'None'}
    clm_lookup['stream'][1] = 'phsen_abcdef_dcl_instrument_recovered'
    clm_lookup['parameters'][2] = {'inp': 'phsen_abcdef_ph_seawater', 'tinp': 'time', 'zinp': 'None'}
    clm_lookup['stream'][2] = 'phsen_abcdef_instrument'

    return annotations, gr_lookup, clm_lookup, clm_table


def main(argv=None):
    """
    Download the PHSEN data from the Gold Copy THREDDS server and create the
    QARTOD gross range and climatology test lookup tables.
    """
    # setup the input arguments
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor

    # create the initial HITL annotation blocks, the QARTOD gross range and climatology lookup values, and
    # the climatology table for the seawater_ph parameter
    annotations, gr_lookup, clm_lookup, clm_table = generate_qartod(site, node, sensor, "2021-01-01T00:00:00")

    # save the resulting annotations and qartod lookups and tables
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/qartod')
    out_path = os.path.abspath(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # save the annotations to a csv file for further processing
    csv_names = ['id', 'subsite', 'node', 'sensor', 'method', 'stream', 'parameters',
                 'beginDate', 'endDate', 'exclusionFlag', 'qcFlag', 'source', 'annotation']
    anno_csv = '-'.join([site, node, sensor]) + '.quality_annotations.csv'
    annotations.to_csv(os.path.join(out_path, anno_csv), index=False, columns=csv_names)

    # save the gross range values to a csv for further processing
    csv_names = ['subsite', 'node', 'sensor', 'stream', 'parameter', 'qcConfig', 'source']
    gr_csv = '-'.join([site, node, sensor]) + '.gross_range.csv'
    gr_lookup.to_csv(os.path.join(out_path, gr_csv), index=False, columns=csv_names)

    # save the climatology values and table to a csv for further processing
    csv_names = ['subsite', 'node', 'sensor', 'stream', 'parameters', 'climatologyTable', 'source']
    clm_csv = '-'.join([site, node, sensor]) + '.climatology.csv'
    clm_tbl = '-'.join([site, node, sensor]) + '-seawater_ph.csv'
    clm_lookup.to_csv(os.path.join(out_path, clm_csv), index=False, columns=csv_names)
    with open(os.path.join(out_path, clm_tbl), 'w') as clm:
        clm.write(clm_table[0])


if __name__ == '__main__':
    main()
