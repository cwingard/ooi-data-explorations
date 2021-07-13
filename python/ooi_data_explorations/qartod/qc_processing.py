#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Calculates QARTOD test ranges and creates the resulting tables used by
    the OOI QC lookup functions to implement the QARTOD testing.
"""
import argparse
import numpy as np
import pandas as pd
import sys

from ooi_data_explorations.qartod.climatology import Climatology
from ooi_data_explorations.qartod.gross_range import GrossRange


def identify_blocks(flags, time_step=None):
    """
    Use a boolean array of quality flags to find and create blocks of data
    bound by the starting and ending dates and times of consecutive flagged
    points. Points are defined as consecutive if they occur within a certain
    time range of the preceding flagged point (default is 8 hours). This helps
    to limit cases of noisy data where the flagging is inconsistent.

    There must be a minimum time range of flagged points (defined as covering
    more than 24 hours as the default) in order to create a block. Consecutive
    blocks must be more than than the minimum time window apart, or they are
    combined into a single block.

    :param flags: a boolean array of data points flagged as failing a QC
        assessment
    :param time_step: a two-value list of the minimum time range to use in
        combining flagged points into a group, and the minimum range of a
        group of flagged points to determine if a block should be created.
        The defaults are 8 and 24 hours, respectively.
    :return blocks: List of starting and ending dates and times defining
        a block of flagged data.
    """
    # find blocks of consecutive points that span a time range greater than time_step[0]
    if time_step is None:
        time_step = [8, 24]
    diff = 0
    flg = False
    dates = []
    start = None
    stop = None
    for i in range(flags.size):
        # index through the boolean array until we find a flagged data point
        if flags.values[i] and not flg:
            diff = 0  # reset the difference estimate
            flg = True  # set the conditional to indicate we have found a bad data point
            start = flags.time.values[i]  # set the start time for the data point

        # if we have identified a starting point and the next value is flagged, set the
        # stop time and calculate the time difference
        if flags.values[i] and flg:
            stop = flags.time.values[i]
            diff = ((stop - start) / 1e9 / 60 / 60).astype(np.int)  # convert from nanoseconds to hours

        # if we have identified a starting point and now find a data point that is not flagged,
        # check to see if either we are at the end of the record or the next set of data points
        # (based on a time window) are flagged. we don't want one good point resetting the block
        # if we have a cluster of good/bad points.
        if not flags.values[i] and flg:
            # check to see if we are at the end of the record
            if i == flags.size:
                stop = flags.time.values[i]
                diff = ((stop - start) / 1e9 / 60 / 60).astype(np.int)  # convert from nanoseconds to hours
                dates.append([start, stop, diff])
                continue

            # look forward time_step[0] hours
            m = (flags.time.values > flags.time.values[i]) & (
                        flags.time.values <= flags.time.values[i] + np.timedelta64(time_step[0], 'h'))

            # if there are bad points within the time window, keep adding them
            if np.any(flags.values[m]):
                stop = flags.time.values[i]
                diff = ((stop - start) / 1e9 / 60 / 60).astype(np.int)  # convert from nanoseconds to hours
            else:
                # otherwise close out the block
                flg = False
                if diff > time_step[0]:
                    dates.append([start, stop, diff])

    # now check the blocks to see if we have any consecutive blocks (less than time_step[1] apart)
    blocks = []
    if dates:
        # first, did we find any blocks of data?
        start = dates[0][0]
        stop = dates[0][1]
        if len(dates) == 1:
            # if there was only one block...
            blocks.append([start, stop])
        else:
            # ...otherwise
            for i in range(1, len(dates)):
                diff = ((dates[i][0] - dates[i - 1][1]) / 1e9 / 60 / 60).astype(np.int)
                # test to see if the difference between blocks is greater than time_step[1]
                if diff > time_step[1]:
                    # create a block
                    stop = dates[i - 1][1]
                    blocks.append([start, stop])
                    # update the start time for the next set
                    start = dates[i][0]

                # test if we are at the end of the blocks, if so use the last point
                if i == len(dates) - 1:
                    stop = dates[i][1]
                    blocks.append([start, stop])

    return blocks


def create_annotations(site, node, sensor, blocks):
    """
    Use the identified blocks of data marked as "fail" to create initial HITL
    annotations for the data. Additional HITL work will be required to review
    the data and the initial annotation flags to create a final HITL set of
    annotations that can be posted to the data base.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :param blocks: Consecutive blocks of bad data determined via the
        identify_blocks function
    :return output: Dictionary of the initial annotations for further review
    """
    # default text to use for the HITL annotation
    fail_text = ('Based on a HITL review and automated quality assessments of the data, the data highlighted '
                 'during this time period is considered inaccurate and users should avoid using the data as '
                 'part of any analysis.')

    # create the initial annotation dictionary structure
    output = {'id': [], 'subsite': [], 'node': [], 'sensor': [], 'method': [], 'stream': [], 'parameters': [],
              'beginDT': [], 'beginDate': [], 'endDT': [], 'endDate': [], 'exclusionFlag': [], 'qcFlag': [],
              'source': [], 'annotation': []}

    for block in blocks:
        start = ((block[0] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64) * 1000
        stop = ((block[1] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64) * 1000
        output['id'].append('')
        output['subsite'].append(site)
        output['node'].append(node)
        output['sensor'].append(sensor)
        output['method'].append(None)
        output['stream'].append(None)
        output['parameters'].append([])
        output['beginDT'].append(start)
        output['beginDate'].append(np.datetime_as_string(block[0], unit='s'))
        output['endDT'].append(stop)
        output['endDate'].append(np.datetime_as_string(block[1], unit='s'))
        output['exclusionFlag'].append('False')
        output['qcFlag'].append('fail')
        output['source'].append('replace.me@whatever.com')
        output['annotation'].append(fail_text)

    return output


def format_climatology(param, clm, sensor_range, site, node, sensor, stream):
    """
    Creates a dictionary object that can later be saved to a CSV formatted
    file for use in the Climatology lookup tables.

    :param param: parameter name of the variable for the calculated climatology
    :param clm: results of the climatology test, used to create the table
    :param sensor_range: list of vendor defined ranges for valid data
    :param site: Site designator, extracted from the first part of the reference
        designator
    :param node: Node designator, extracted from the second part of the reference
        designator
    :param sensor: Sensor designator, extracted from the third and fourth part of
        the reference designator
    :param stream: Stream name that contains the data of interest
    :return qc_dict: dictionary with the sensor and user gross range values
        added in the formatting expected by the QC lookup
    """
    # create the lookup dictionary
    var_explained = clm.regression['variance_explained']
    qc_dict = {
        'subsite': site,
        'node': node,
        'sensor': sensor,
        'stream': stream,
        'parameters': {'inp': param, 'tinp': 'time', 'zinp': 'None'},
        'climatologyTable': 'climatology_tables/{}-{}-{}-{}.csv'.format(site, node, sensor, param),
        'source': 'The variance explained by the climatological model is {:.1%}.'.format(var_explained[0])
    }

    # create the climatology table
    header_str = ''
    value_str = '[0, 0]'
    for idx, mu in enumerate(clm.monthly_fit):
        # use the index number to create the header row
        header_str += ',[{}, {}]'.format(idx+1, idx+1)

        # calculate the climatological ranges
        cmin = mu - clm.monthly_std.values[idx] * 3
        if cmin < sensor_range[0]:
            cmin = sensor_range[0]

        cmax = mu + clm.monthly_std.values[idx] * 3
        if cmax > sensor_range[1]:
            cmax = sensor_range[1]

        # append the data to ranges
        value_str += ',[{:.2f}, {:.2f}]'.format(cmin, cmax)

    clm_table = header_str + '\n' + value_str

    return qc_dict, clm_table


def process_climatology(ds, params, sensor_range, **kwargs):
    """
    Using the data in an xarray dataset and a list of parameter(s), calculate
    a monthly climatology for each parameter and create formatted outputs
    that can be saved to the qc_lookup tables used by OOI for QARTOD testing

    :param ds: dataset with the parameter(s) to use in developing a climatology
    :param params: list of the parameter name(s) of the variable(s) used for
        the calculated climatology
    :param sensor_range: list of vendor defined ranges for valid data
    :keyword site: Site designator, extracted from the first part of the
        reference designator (optional input)
    :keyword node: Node designator, extracted from the second part of the
        reference designator (optional input)
    :keyword sensor: Sensor designator, extracted from the third and fourth
        part of the reference designator (optional input)
    :keyword stream: Stream name that contains the data of interest
        (optional input)
    :return clm_lookup: a pandas Dataframe corresponding to the values used to
        create the QARTOD lookup tables for the climatology test
    :return clm_tables: list of formatted strings containing the values used to
        define the monthly climatology test limits
    """
    # process the optional keyword arguments
    site = kwargs.get('site')
    node = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')

    # initialize the Climatology class
    clm = Climatology()

    # create an empty panda dataframe and list to hold the results
    clm_lookup = pd.DataFrame()
    clm_tables = []

    # loop through the parameter(s) of interest
    sensor_range = np.atleast_2d(sensor_range).tolist()
    for idx, param in enumerate(params):
        # calculate the 2-cycle climatology for the parameter of interest
        clm.fit(ds, param)

        # create the formatted dictionary for the lookup tables
        qc_dict, clm_table = format_climatology(param, clm, sensor_range[idx], site, node, sensor, stream)

        # append the dictionary to the dataframe and the table to the list
        clm_lookup = clm_lookup.append(qc_dict, ignore_index=True)
        clm_tables.append(clm_table)

    # return the results
    return clm_lookup, clm_tables


def format_gross_range(param, sensor_range, user_range, site, node, sensor, stream, source):
    """
    Creates a dictionary object that can later be saved to a CSV formatted
    file for use in the Gross Range lookup tables.

    :param param: parameter name of the variable for the calculated user range
    :param sensor_range: default sensor, or fail range, usually referenced
        from the vendor documentation
    :param user_range: user range, or sensor range, calculated from the data
    :param site: Site designator, extracted from the first part of the reference
        designator
    :param node: Node designator, extracted from the second part of the reference
        designator
    :param sensor: Sensor designator, extracted from the third and fourth part of
        the reference designator
    :param stream: Stream name that contains the data of interest
    :param source: Notes or comments about how the Gross Range values were
        obtained
    :return qc_dict: dictionary with the sensor and user gross range values
        added in the formatting expected by the QC lookup tables
    """
    # create the dictionary
    qc_dict = {
        'subsite': site,
        'node': node,
        'sensor': sensor,
        'stream': stream,
        'parameter': {
            'inp': param
        },
        'qcConfig': {
             'qartod': {
                 'gross_range_test': {
                     'suspect_span': ['{}'.format(user_range[0]), '{}'.format(user_range[1])],
                     'fail_span': ['{}'.format(sensor_range[0]), '{}'.format(sensor_range[1])]
                 }
             }
         },
        'source': source
    }
    return qc_dict


def process_gross_range(ds, params, sensor_range, **kwargs):
    """
    Using the data in an xarray dataset and a list of parameter(s), calculate
    a gross ranges (long term average plus/minus 3 standard deviations) for
    each parameter and create formatted outputs that can be saved to the
    qc_lookup tables used by OOI for QARTOD testing.

    :param ds: dataset with the parameter(s) to use in developing a gross range
    :param params: list of the parameter name(s) of the variable(s) used for
        the calculated user portion of the gross range test
    :param sensor_range: list of vendor defined ranges for valid data
    :keyword site: Site designator, extracted from the first part of the
        reference designator (optional input)
    :keyword node: Node designator, extracted from the second part of the
        reference designator (optional input)
    :keyword sensor: Sensor designator, extracted from the third and fourth
        part of the reference designator (optional input)
    :keyword stream: Stream name that contains the data of interest
        (optional input)
    :return gross_range: a pandas Dataframe corresponding to the values used to
        create the QARTOD lookup tables for the gross range test
    """
    # process the optional keyword arguments
    site = kwargs.get('site')
    node = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')
    source = kwargs.get('source')

    # create an empty pandas dataframe to hold the results
    gross_range = pd.DataFrame()

    # loop through the parameter(s) of interest
    sensor_range = np.atleast_2d(sensor_range).tolist()
    for idx, param in enumerate(params):
        # calculate the user range
        gr = GrossRange(fail_min=sensor_range[idx][0], fail_max=sensor_range[idx][1])
        gr.fit(ds, param, 3)
        user_range = [gr.suspect_min, gr.suspect_max]
        # create the formatted dictionary
        qc_dict = format_gross_range(param, sensor_range[idx], user_range, site, node, sensor, stream, source)
        # append the dictionary to the dataframe
        gross_range = gross_range.append(qc_dict, ignore_index=True)

    # return the results
    return gross_range


def inputs(argv=None):
    """
    Sets the main input arguments that will be used in the QC processing
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize argument parser
    parser = argparse.ArgumentParser(
        description="""Download and process instrument data to generate QARTOD lookup tables""")

    # assign input arguments.
    parser.add_argument("-s", "--site", dest="site", type=str, required=True)
    parser.add_argument("-n", "--node", dest="node", type=str, required=True)
    parser.add_argument("-sn", "--sensor", dest="sensor", type=str, required=True)

    # parse the input arguments and create a parser object
    args = parser.parse_args(argv)

    return args
