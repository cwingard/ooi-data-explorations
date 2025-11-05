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

from scipy.stats import normaltest
from scipy.stats import median_abs_deviation as mad
import dask
from dask.diagnostics import ProgressBar

from ooi_data_explorations.qartod.climatology import Climatology

# csv file ordered header row
ANNO_HEADER = ['id', 'subsite', 'node', 'sensor', 'stream', 'method', 'parameters',
               'beginDate', 'endDate', 'exclusionFlag', 'qcFlag', 'source', 'annotation']
CLM_HEADER = ['subsite', 'node', 'sensor', 'stream', 'parameters', 'climatologyTable', 'source', 'notes']
GR_HEADER = ['subsite', 'node', 'sensor', 'stream', 'parameter', 'qcConfig', 'source', 'notes']


def woa_standard_bins():
    """
    Construct a 2D array of the World Ocean Atlas (WOA) Standard Depth Bins
    from the surface to 5500 meters for use in the Climatology calculations.
    The bins are used to select data based on depth that fall between 2
    standard depth levels (e.g. between 0 m (Level 1) and 5 m (Level 2)).

    Further information on the WOA depth bins is available in the documentation
    available from the National Centers for Environmental Information (NCEI):
    https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DOC/woa18documentation.pdf

    :return: The WOA standard depth bins as a 2D numpy array
    """
    # Surface to 100 m by 5 m
    left = np.atleast_2d(np.arange(0, 100, 5)).T
    right = np.atleast_2d(np.arange(5, 105, 5)).T
    m100 = np.concatenate((left, right), axis=1)

    # 100 m to 500 m by 25 m
    left = np.atleast_2d(np.arange(100, 500, 25)).T
    right = np.atleast_2d(np.arange(125, 525, 25)).T
    m500 = np.concatenate((left, right), axis=1)

    # 500 m to 2000 m by 50 m
    left = np.atleast_2d(np.arange(500, 2000, 50)).T
    right = np.atleast_2d(np.arange(550, 2050, 50)).T
    m2000 = np.concatenate((left, right), axis=1)

    # 2000 m to 5500 m by 1000 m
    left = np.atleast_2d(np.arange(2000, 5500, 100)).T
    right = np.atleast_2d(np.arange(2100, 5600, 100)).T
    m5500 = np.concatenate((left, right), axis=1)

    woa_bins = np.concatenate((m100, m500, m2000, m5500), axis=0)
    return woa_bins


def identify_blocks(flags, time_step, padding=0):
    """
    Use a boolean array of quality flags to find and create blocks of data
    bound by the starting and ending dates and times of consecutive flagged
    points. Points are defined as consecutive if they occur within a certain
    time range of the preceding flagged point (default is 8 hours). This helps
    to limit cases of noisy data where the flagging is inconsistent.

    There must be a minimum time range of flagged points (defined as covering
    more than 24 hours as the default) in order to create a block. Consecutive
    blocks must be more than the minimum time window apart, or they are
    combined into a single block.

    :param flags: a boolean array of data points flagged as failing a QC
        assessment
    :param time_step: a two-value list of the minimum time range to use in
        combining flagged points into a group, and the minimum range of a
        group of flagged points to determine if a block should be created.
    :param padding: add padding (in hours) to identified blocks
    :return: List of starting and ending dates and times defining
        a block of flagged data.
    """
    # find blocks of consecutive points that span a time range greater than time_step[0]
    padding = np.timedelta64(padding, 'h')
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
            diff = ((stop - start) / 1e9 / 60 / 60).astype(int)  # convert from nanoseconds to hours

        # if we have identified a starting point and now find a data point that is not flagged,
        # check to see if either we are at the end of the record or the next set of data points
        # (based on a time window) are flagged. we don't want one good point resetting the block
        # if we have a cluster of good/bad points.
        if not flags.values[i] and flg:
            # check to see if we are at the end of the record
            if i == flags.size:
                stop = flags.time.values[i]
                diff = ((stop - start) / 1e9 / 60 / 60).astype(int)  # convert from nanoseconds to hours
                dates.append([start, stop, diff])
                continue

            # look forward time_step[0] hours
            m = (flags.time.values > flags.time.values[i]) & (
                        flags.time.values <= flags.time.values[i] + np.timedelta64(time_step[0], 'h'))

            # if there are bad points within the time window, keep adding them
            if np.any(flags.values[m]):
                stop = flags.time.values[i]
                diff = ((stop - start) / 1e9 / 60 / 60).astype(int)  # convert from nanoseconds to hours
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
            blocks.append([start - padding, stop + padding])
        else:
            # ...otherwise
            for i in range(1, len(dates)):
                diff = ((dates[i][0] - dates[i - 1][1]) / 1e9 / 60 / 60).astype(int)
                # test to see if the difference between blocks is greater than time_step[1]
                if diff > time_step[1]:
                    # create a block
                    stop = dates[i - 1][1]
                    blocks.append([start - padding, stop + padding])
                    # update the start time for the next set
                    start = dates[i][0]

                # test if we are at the end of the blocks, if so use the last point
                if i == len(dates) - 1:
                    stop = dates[i][1]
                    blocks.append([start - padding, stop + padding])

    return blocks


def create_annotations(site, node, sensor, blocks):
    """
    Use the identified blocks of data marked as "fail" to create initial HITL
    annotations for the data. Additional HITL work will be required to review
    the data and the initial annotation flags to create a final HITL set of
    annotations that can be posted to the database.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :param blocks: Consecutive blocks of bad data determined via the
        identify_blocks function
    :return: Dictionary of the initial annotations for further review
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


def format_climatology(parameter, clm, sensor_range, depth_bins, site, node, sensor, stream,
                       fixed_lower, fixed_upper, stdx=3):
    """
    Creates a dictionary object that can later be saved to a CSV formatted
    file for use in the Climatology lookup tables.

    :param parameter: parameter name of the variable for the calculated climatology
    :param clm: results of the climatology test, used to create the table
    :param sensor_range: list of vendor defined ranges for valid data
    :param depth_bins: depth bins used for the climatology
    :param site: Site designator, extracted from the first part of the reference
        designator
    :param node: Node designator, extracted from the second part of the reference
        designator
    :param sensor: Sensor designator, extracted from the third and fourth part of
        the reference designator
    :param stream: Stream name that contains the data of interest
    :param fixed_lower: fixed, pre-defined value to use for the lower range
        limit of the climatology test
    :param fixed_upper: fixed, pre-defined value to use for the upper range
        limit of the climatology test
    :param stdx: number of standard deviations to use in the climatology test
    :return: dictionary with the sensor and user gross range values
        added in the formatting expected by the QC lookup
    """
    # set up the depth bins, if set
    header_str = ''
    if depth_bins.any():
        value_str = '"[{}, {}]"'.format(depth_bins[0], depth_bins[1])
        source = 'Climatology based on depth bins (from {} to {} m).'.format(depth_bins[0], depth_bins[1])
    else:
        value_str = '"[0, 0]"'
        source = ''

    # create the lookup dictionary
    qc_dict = {
        'subsite': site,
        'node': node,
        'sensor': sensor,
        'stream': stream,
        'parameters': {'inp': parameter, 'tinp': 'time', 'zinp': 'None'},
        'climatologyTable': 'climatology_tables/{}-{}-{}-{}.csv'.format(site, node, sensor, parameter),
        'source': source,
    }

    # pull out the variance explained by the climatology model and set to 0 if not available
    var_explained = clm.regression['variance_explained']
    if len(var_explained) == 0:
        var_explained = [0]

    # create the climatology table
    for idx, mu in enumerate(clm.monthly_fit):
        # use the index number to create the header row
        header_str += ',"[{}, {}]"'.format(idx + 1, idx + 1)

        # calculate the climatological ranges
        cmin = mu - clm.monthly_std.values[idx] * stdx
        if fixed_lower or (cmin < sensor_range[0] or cmin > sensor_range[1]):
            cmin = sensor_range[0]

        cmax = mu + clm.monthly_std.values[idx] * stdx
        if fixed_upper or (cmax > sensor_range[1] or cmax < sensor_range[0]):
            cmax = sensor_range[1]

        # append the data to ranges
        value_str += ',"[{:.5f}, {:.5f}]"'.format(cmin, cmax)

        # update the notes
        if var_explained[0] < 0.15:
            qc_dict['notes'] = ('The climatological ranges are based on the monthly mean plus/minus {}x '
                                'the monthly standard deviations.'.format(stdx))
        else:
            qc_dict['notes'] = ('The climatological ranges are based on a 2-cycle harmonic fit to the monthly means '
                                'plus/minus {}x the monthly standard deviations. The variance explained by the '
                                'climatological model is {:.1%}.'.format(stdx, var_explained[0]))

    clm_table = header_str + '\n' + value_str

    return qc_dict, clm_table


def process_climatology(ds, parameters, sensor_range, **kwargs):
    """
    Using the data in a xarray dataset and a list of parameter(s), calculate
    a monthly climatology for each parameter and create formatted outputs
    that can be saved to the qc_lookup tables used by OOI for QARTOD testing

    :param ds: dataset with the parameter(s) to use in developing a climatology
    :param parameters: list of the parameter name(s) of the variable(s) used for
        the calculated climatology
    :param sensor_range: list of vendor defined ranges for valid data
    :**depth_bins: 2D array of WOA depth bins
    :**fixed_lower: boolean flag to set the lower range to the sensor range
        (optional input)
    :**fixed_upper: boolean flag to set the upper range to the sensor range
        (optional input)
    :**stdx: use Nx the standard deviation for the user range (optional input,
        default is 3x)
    :**site: Site designator, extracted from the first part of the
        reference designator (optional input)
    :**node: Node designator, extracted from the second part of the
        reference designator (optional input)
    :**sensor: Sensor designator, extracted from the third and fourth
        part of the reference designator (optional input)
    :**stream: Stream name that contains the data of interest
        (optional input)
    :return clm_lookup: a pandas Dataframe corresponding to the values used to
        create the QARTOD lookup tables for the climatology test
    :return clm_tables: list of formatted strings containing the values used to
        define the monthly climatology test limits
    """
    # process the optional keyword arguments
    depth_bins = kwargs.get('depth_bins')
    fixed_lower = kwargs.get('fixed_lower')
    fixed_upper = kwargs.get('fixed_upper')
    stdx = kwargs.get('stdx', 3)
    site = kwargs.get('site')
    node = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')

    # initialize the Climatology class
    clm = Climatology()

    # create an empty panda dataframe and list to hold the results
    clm_lookup = []
    clm_tables = []

    # check the type of the depth bins, and set to an empty array if NoneType
    if isinstance(depth_bins, type(None)):
        depth_bins = np.array([])

    # loop through the parameter(s) of interest
    sensor_range = np.atleast_2d(sensor_range).tolist()
    for idx, param in enumerate(parameters):
        if param in ds.variables:
            if depth_bins.any():
                depth_tables = ''
                for bins in depth_bins:
                    # slice the dataset, selecting our data based on depth ranges
                    sliced = ds[param].where((ds.depth >= bins[0]) & (ds.depth <= bins[1]), drop=True).to_dataset()
                    if len(sliced[param]) == 0:
                        continue

                    # sort based on time and make sure we have a monotonic dataset
                    sliced = sliced.sortby('time')
                    _, index = np.unique(sliced['time'], return_index=True)
                    sliced = sliced.isel(time=index)

                    # calculate the 2-cycle climatology for the parameter of interest
                    m = (sliced[param] > sensor_range[idx][0]) & (sliced[param] < sensor_range[idx][1]) \
                        & (~np.isnan(sliced[param]))
                    sliced = sliced[param].where(m, drop=True)
                    clm.fit(sliced)

                    # create the formatted dictionary for the lookup tables
                    qc_dict, clm_table = format_climatology(param, clm, sensor_range[idx], bins, site, node, sensor,
                                                            stream, fixed_lower, fixed_upper, stdx)

                    # append the dictionary to the dataframe and build the depth table
                    df = (pd.Series(qc_dict).to_frame()).transpose()
                    clm_lookup.append(df)
                    if depth_tables:
                        depth_tables += clm_table[114:]
                    else:
                        depth_tables += clm_table

                # add the final depth table for the parameter
                clm_tables.append(depth_tables)
            else:
                # calculate the 2-cycle climatology for the parameter of interest
                m = (ds[param] > sensor_range[idx][0]) & (ds[param] < sensor_range[idx][1]) & (~np.isnan(ds[param]))
                da = ds[param].where(m, drop=True)
                clm.fit(da)

                # create the formatted dictionary for the lookup tables
                qc_dict, clm_table = format_climatology(param, clm, sensor_range[idx], depth_bins,
                                                        site, node, sensor, stream, fixed_lower, fixed_upper)

                # append the dictionary to the dataframe and the table to the list
                df = (pd.Series(qc_dict).to_frame()).transpose()
                clm_lookup.append(df)
                clm_tables.append(clm_table)

    # return the results
    clm_lookup = pd.concat(clm_lookup, ignore_index=True, sort=False)
    return clm_lookup, clm_tables


def format_gross_range(parameter, sensor_range, user_range, site, node, sensor, stream, notes):
    """
    Creates a dictionary object that can later be saved to a CSV formatted
    file for use in the Gross Range lookup tables.

    :param parameter: parameter name of the variable for the calculated user range
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
    :param notes: Notes or comments about how the Gross Range values were
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
            'inp': parameter
        },
        'qcConfig': {
             'qartod': {
                 'gross_range_test': {
                     'suspect_span': user_range,
                     'fail_span': sensor_range
                 }
             }
         },
        'source': '',
        'notes': notes
    }
    return qc_dict


def process_gross_range(ds, parameters, sensor_range, **kwargs):
    """
    Using the data in a xarray dataset and a list of parameter(s), calculate
    a gross ranges (long term average plus/minus 3 standard deviations) for
    each parameter and create formatted outputs that can be saved to the
    qc_lookup tables used by OOI for QARTOD testing. Exceptions are made for
    parameters that are not normally distributed, where the user range is
    based on the median and median absolute deviation. Additionally, the user
    range can be expanded to 5 times the standard deviation or median absolute
    deviation.

    :param ds: dataset with the parameter(s) to use in developing a gross range
    :param parameters: list of the parameter name(s) of the variable(s) used for
        the calculated user portion of the gross range test
    :param sensor_range: list of vendor defined ranges for valid data
    :**fixed_lower: boolean flag to set the lower range to the sensor range
        (optional input)
    :**fixed_upper: boolean flag to set the upper range to the sensor range
        (optional input)
    :**stdx: use Nx the standard deviation for the user range (optional input,
        default is 3x)
    :**site: Site designator, extracted from the first part of the
        reference designator (optional input)
    :**node: Node designator, extracted from the second part of the
        reference designator (optional input)
    :**sensor: Sensor designator, extracted from the third and fourth
        part of the reference designator (optional input)
    :**stream: Stream name that contains the data of interest
        (optional input)
    :return: a pandas Dataframe corresponding to the values used to create the
        QARTOD lookup tables for the gross range test
    """
    # process the optional keyword arguments
    fixed_lower = kwargs.get('fixed_lower')
    fixed_upper = kwargs.get('fixed_upper')
    stdx = kwargs.get('stdx', 3)
    site = kwargs.get('site')
    node = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')

    if stdx == 3:
        percents = [0.15, 99.85]
    else:
        percents = [0.0115, 99.9885]

    # create an empty pandas dataframe to hold the results
    gross_range = []

    # loop through the parameter(s) of interest, roughly estimating if the data is normally distributed using a
    # bootstrap analysis to randomly select 4500 data points to use, running the test a total of 5000 times
    sensor_range = np.atleast_2d(sensor_range).tolist()
    for idx, param in enumerate(parameters):
        if param in ds.variables:
            # Select out the data array of the desired param. This speeds up the processing
            m = (ds[param] > sensor_range[idx][0]) & (ds[param] < sensor_range[idx][1]) & (~np.isnan(ds[param]))
            da = ds[param][m]

            # Utilize dask to parallelize the random choice and calculate the pnorm
            random_choice = dask.delayed(np.random.choice)

            # Select out the dataarray of the desired param. This speeds up the process
            m = (ds[param] > sensor_range[idx][0]) & (ds[param] < sensor_range[idx][1]) & (~np.isnan(ds[param]))
            da = ds[param].where(m, drop=True).values.flatten()
            vals = []
            for i in range(5000):
                vals.append(random_choice(da, 4500))

            # reset the dataarray
            da = ds[param].where(m, drop=True)

            # Now compute the pnorm values via dask.delayed
            with ProgressBar():
                print("Testing data for normality: %s" % param)
                pvals = dask.compute(*vals)

            pnorm = [normaltest(v).pvalue for v in pvals]
            if np.mean(pnorm) < 0.05:
                # Even with a log-normal transformation, the data is not normally distributed, so we will
                # set the user range using percentiles that approximate the Empirical Rule, covering
                # 99.7% of the data
                lower = np.nanpercentile(da, percents[0])
                upper = np.nanpercentile(da, percents[1])
                if stdx == 3:
                    notes = ('User range based on percentiles of the observations, which are not normally '
                             'distributed. Percentiles were chosen to cover 99.7% of the data, approximating the '
                             'Empirical Rule.')
                else:
                    notes = ('User range based on percentiles of the observations, which are not normally '
                             'distributed. Percentiles were chosen to cover 99.977% of the data, approximating '
                             'a 5-sigma range.')
            else:
                # most likely this data is normally distributed, or close enough, and we can use the mean
                mu = da.mean().values
                sd = da.std().values
                lower = mu - sd * stdx
                upper = mu + sd * stdx
                notes = 'User range based on the mean +- {} standard deviations of all observations.'.format(stdx)

            # reset the lower and upper ranges if they exceed the sensor ranges
            if fixed_lower or lower < sensor_range[idx][0]:
                lower = sensor_range[idx][0]

            if fixed_upper or upper > sensor_range[idx][1]:
                upper = sensor_range[idx][1]

            # create the formatted dictionary
            user_range = [np.round(lower, decimals=5), np.round(upper, decimals=5)]
            qc_dict = format_gross_range(param, sensor_range[idx], user_range, site, node, sensor, stream, notes)

            # append the dictionary to the dataframe
            df = (pd.Series(qc_dict).to_frame()).transpose()
            gross_range.append(df)

    # return the results
    gross_range = pd.concat(gross_range, ignore_index=True, sort=False)
    return gross_range


def parse_qc(ds):
    """
    Extract the QC test results from the different variables in the data set,
    and create a new variable with the QC test results set to match the logic
    used in QARTOD testing. Instead of setting the results to an integer
    representation of a bitmask, use the pass = 1, not_evaluated = 2,
    suspect_or_of_high_interest = 3, fail = 4 and missing = 9 flag values from
    QARTOD.

    This code was inspired by an example notebook developed by the OOI Data
    Team for the 2018 Data Workshops. The original example, by Friedrich Knuth,
    and additional information on the original OOI QC algorithms can be found
    at:

    https://oceanobservatories.org/knowledgebase/interpreting-qc-variables-and-results/

    :param ds: dataset with *_qc_executed and *_qc_results variables
    :return ds: dataset with the *_qc_executed and *_qc_results variables
        reworked to create a new *_qc_summary variable with the results
        of the QC checks decoded into a QARTOD style flag value.
    """
    # create a list of the variables that have had QC tests applied
    variables = [x.split('_qc_results')[0] for x in ds.variables if 'qc_results' in x]

    # for each variable with qc tests applied
    for var in variables:
        # set the qc_results and qc_executed variable names and the new qc_flags variable name
        qc_result = var + '_qc_results'
        qc_executed = var + '_qc_executed'
        qc_summary = var + '_qc_summary_flag'

        # create the initial qc_flags array
        flags = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0]), (len(ds.time), 1))
        # the list of tests run, and their bit positions are:
        #    0: dataqc_globalrangetest
        #    1: dataqc_localrangetest
        #    2: dataqc_spiketest
        #    3: dataqc_polytrendtest
        #    4: dataqc_stuckvaluetest
        #    5: dataqc_gradienttest
        #    6: undefined
        #    7: dataqc_propagateflags

        # use the qc_executed variable to determine which tests were run, and set up a bit mask to pull out the results
        executed = np.bitwise_or.reduce(ds[qc_executed].values.astype('uint8'))
        executed_bits = np.unpackbits(executed.astype('uint8'))

        # for each test executed, reset the qc_flags for pass == 1, suspect == 3, or fail == 4
        for index, value in enumerate(executed_bits[::-1]):
            if value:
                if index in [2, 3, 4, 5, 6, 7]:
                    flag = 3
                else:
                    # only mark the global range test as fail, all the other tests are problematic
                    flag = 4
                mask = 2 ** index
                m = (ds[qc_result].values.astype('uint8') & mask) > 0
                flags[m, index] = 1   # True == pass
                flags[~m, index] = flag  # False == suspect/fail

        # add the qc_flags to the dataset, rolling up the results into a single value
        ds[qc_summary] = ('time', flags.max(axis=1, initial=1).astype(np.int32))

        # set up the attributes for the new variable
        ds[qc_summary].attrs = dict({
            'long_name': '%s QC Summary Flag' % ds[var].attrs['long_name'],
            'standard_name': 'aggregate_quality_flag',
            'comment': ('Converts the QC Results values from a bitmap to a QARTOD style summary flag, where '
                        'the values are 1 == pass, 2 == not evaluated, 3 == suspect or of high interest, '
                        '4 == fail, and 9 == missing. The QC tests, as applied by OOI, only yield pass or '
                        'fail values.'),
            'flag_values': np.array([1, 2, 3, 4, 9]).astype(np.int32),
            'flag_meanings': 'pass not_evaluated suspect_or_of_high_interest fail missing'
        })

    return ds


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
    parser.add_argument("-co", "--cut_off", dest="cut_off", type=str, required=False)
    parser.add_argument("-dp", "--deploy", dest="deployment", type=int, required=False)
    parser.add_argument("-cn", "--cruise", dest="cruise", type=str, required=False)

    # parse the input arguments and create a parser object
    args = parser.parse_args(argv)

    return args
