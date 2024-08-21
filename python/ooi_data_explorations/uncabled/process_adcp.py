import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

# Can use these based on NDBC but recommend using the 
# TRDI QAQC Model rev12-1
QCThresholds = {
    'error_velocity': {
        'pass': 0.05,
        'fail': 0.20 },
    'correlation_magnitude': {
        'pass': 115,
        'fail': 63 },
    'vertical_velocity': {
        'pass': 0.30,
        'fail': 0.50 },
    'horizontal_speed': {
        'pass': 1.00,
        'fail': 2.50 },
    'percent_good': {
        'ADCPT': {
            'pass': 56,
            'fail': 45 },
        'ADCPS': {
            'pass': 48,
            'fail': 38 }
    }
}


def sidelobe_depth(ds: xr.Dataset, theta: int = 20) -> xr.DataArray:
    """
    Calculate the sidelobe contamination depth for the given ADCP.

    The sidelobe intereference depth 
    is caluclated following Lentz et al. (2022) where:
    
        z_ic = ha*[1 - cos(theta)] + 3*delta_Z/2
        
    z_ic is the depth above which there is sidelobe interference,
    ha is the transducer face depth, theta is the beam angle, and
    delta_Z is the cell-bin depth. We ignore instrument tilt at 
    this time and its impact, assuming fixed beam angle.

    Parameters
    ----------
    ds: xarray.dataset
        A TRDI ADCP dataset from OOI from which to calculate the
        sidelobe interference depth
    theta: float, Default = 20
        Beam angle of the given ADCP

    Returns
    -------
    z_ic:
    """
    # First, get the transducer depth
    depth = ds['depth_from_pressure']
    ha = depth.interpolate_na(dim='time', method='linear')

    # Next, get the beam angle
    theta = np.deg2rad(theta)

    # Grab the cell length and convert to m
    delta_z = ds['cell_length'].mean(skipna=True)/100

    # Calculate the range of cells contaminated by sidelobe interference
    z_ic = ha*(1 - np.cos(theta)) + 3*delta_z/2

    return z_ic


def sidelobe_qc(ds: xr.Dataset) -> xr.Dataset:
    """
    Add sidelobe interference QARTOD-style flags to the ADCP dataset

    Assignment of QARTOD style quality flags to the ADCP velocity data based
    on estimation of sidelobe contamination. The sidelobe intereference depth 
    is calculated following Lentz et al. (2022) where:
    
        z_ic = ha*[1 - cos(theta)] + 3*delta_Z/2
        
    z_ic is the depth above which there is sidelobe interference,
    ha is the transducer face depth, theta is the beam angle, and
    delta_Z is the cell-bin depth. Bin depths less than z_ic are
    considered contaminated and the data is considered bad. The assigned flag
    values are:
    
        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail
        9 = Missing

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the TRDI ADCP data downloaded from
        OOINet in .netcdf format

    Returns
    -------
    ds: xarray.Dataset
        The input dataset with the sidelobe interference QC flag
        added as a dataset variable
    """
    # First, get the sidelobe contamination depth z_ic
    z_ic = sidelobe_depth(ds)

    # Next, create a qc_flag for each bin measurement and
    # identify the bins which are shallower than z_ic
    qc_flag = np.ones(ds['bin_depths'].shape, dtype=int)
    mask = ds['bin_depths'] < z_ic
    qc_flag[mask] = 4

    # Add the qc_flags
    qc_name = 'bin_depths_qc_summary_flag'
    ds[qc_name] =  (['time', 'bins'], qc_flag)
    
     # set up the attributes for the new variable
    ds[qc_name].attrs = dict({
        'long_name': '%s QC Summary Flag' % ds['bin_depths'].attrs['long_name'],
        'comment': ('A QARTOD style summary flag indicating depth bins with sidelobe contamination, where ',
                    'the values are 1 == pass, 2 == not evaluated, 3 == suspect or of high interest, ',
                    '4 == fail, and 9 == missing. The QC tests, as applied by OOI, only yield pass or ',
                    'fail values. Sidelobe contamination depth is determined following Lentz et al (2022).'),
        'flag_values': np.array([1, 2, 3, 4, 9]),
        'flag_meanings': 'pass not_evaluated suspect_or_of_high_interest fail missing'
    })

    return ds


def error_velocity_qc(ds: xr.Dataset, suspect: float | int, fail: float | int) -> npt.NDArray[int]:
    """
    Determine ADCP QC based on Error velocity and assign
    QARTOD-style flags. This algorithm uses thresholds computed
    using the TRDI ADCP Data AQ-QC Model rev12-1. The assigned 
    flag values are:

        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail

    The pass, suspect, fail flags are defined as follows:
        pass: error velocity is less than the suspect threshold
        suspect: error velocity is between the suspect and fail thresholds 
        fail: error velocity exceeds the fail threshold

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the TRDI ADCP data downloaded from
        OOINet in .netcdf format
    suspect: float
        The suspect threshold computed from the TRDI QA-QC Model
    fail: flot
        The fail threshold computed from the TRDI QA-QC Model

    Returns
    -------
    qc_flags: numpy.array[int]
        An array of QARTOD-style flags indicating the results of the QC
        test for each given datum

    """
    # Set up a qc_flags the shape of the variable
    qc_flags = np.ones(ds['error_seawater_velocity'].shape, dtype=int)
    
    # Now find the "suspect" values
    mask = (np.abs(ds['error_seawater_velocity']) > suspect)
    qc_flags[mask] = 3
    
    # Now find the "fail" values
    mask = (np.abs(ds['error_seawater_velocity']) > fail)
    qc_flags[mask] = 4

    return qc_flags


def vertical_velocity_qc(ds: xr.Dataset, suspect: float | int, fail: float | int) -> npt.NDArray[int]:
    """
    Determine ADCP QC based on vertical velocity and assign
    QARTOD-style flags. This algorithm uses thresholds computed
    using the TRDI ADCP Data AQ-QC Model rev12-1. The assigned 
    flag values are:

        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail

    The pass, suspect, fail flags are defined as follows:
        pass: vertical velocity is less than the suspect threshold
        suspect: vertical velocity is between the suspect and fail thresholds 
        fail: vertical velocity exceeds the fail threshold

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the TRDI ADCP data downloaded from
        OOINet in .netcdf format
    suspect: float
        The suspect threshold computed from the TRDI QA-QC Model
    fail: flot
        The fail threshold computed from the TRDI QA-QC Model

    Returns
    -------
    qc_flags: numpy.array[int]
        An array of QARTOD-style flags indicating the results of the QC
        test for each given datum
    """
    # Set up a qc_flags the shape of the variable
    qc_flags = np.ones(ds['upward_seawater_velocity'].shape, dtype=int)
    
    # Now find the "suspect" values
    mask = (np.abs(ds['upward_seawater_velocity']) > suspect)
    qc_flags[mask] = 3
    
    # Now find the "fail" values
    mask = (np.abs(ds['upward_seawater_velocity']) > fail)
    qc_flags[mask] = 4

    return qc_flags


def horizontal_speed_qc(ds: xr.Dataset, suspect: float, fail: float) -> npt.NDArray[int]:
    """
    Determine ADCP QC based on vertical velocity and assign
    QARTOD-style flags. This algorithm uses thresholds computed
    using the TRDI ADCP Data AQ-QC Model rev12-1. The assigned 
    flag values are:

        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail

    The pass, suspect, fail flags are defined as follows:
        pass: both east and north velocities are good OR one is suspect
              and the other is good
        suspect: both east and north velocities are suspect
        fail: either east or north velocities fail

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the TRDI ADCP data downloaded from
        OOINet in .netcdf format
    suspect: float
        The suspect threshold computed from the TRDI QA-QC Model
    fail: flot
        The fail threshold computed from the TRDI QA-QC Model

    Returns
    -------
    qc_flags: numpy.array[int]
        An array of QARTOD-style flags indicating the results of the QC
        test for each given datum
    """
    # Set up a qc_flags the shape of the variable
    qc_flags_east = np.ones(ds['eastward_seawater_velocity'].shape, dtype=int)
    qc_flags_north = np.ones(ds['northward_seawater_velocity'].shape, dtype=int)
    
    # Now find the "suspect" values
    mask = (np.abs(ds['eastward_seawater_velocity']) > suspect)
    qc_flags_east[mask] = 3
    mask = (np.abs(ds['northward_seawater_velocity']) > suspect)
    qc_flags_north[mask] = 3
    
    # Now find the "fail" values
    mask = (np.abs(ds['eastward_seawater_velocity']) > fail)
    qc_flags_east[mask] = 4
    mask = (np.abs(ds['northward_seawater_velocity']) > fail)
    qc_flags_north[mask] = 4

    # Now combine them using math to parse out the combinations
    qc_flags = np.ones(ds['eastward_seawater_velocity'].shape, dtype=int)
    
    # Suspect flags when both directions are suspect
    suspect_flags = ((qc_flags_east == 3) & (qc_flags_north == 3))
    qc_flags[suspect_flags] = 3

    # Fail flags when either direction is bad
    bad_flags = ((qc_flags_east == 4) | (qc_flags_north == 4))
    qc_flags[bad_flags] = 4

    return qc_flags


def correlation_magnitude_qc(ds: xr.Dataset, suspect: float, fail: float) -> npt.NDArray[int]:
    """
    Determine ADCP QC based on correlation magnitude and assign
    QARTOD-style flags. This algorithm uses thresholds computed
    using the TRDI ADCP Data AQ-QC Model rev12-1. The assigned 
    flag values are:

        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail

    The pass, suspect, fail are defined as follows:
        pass: correlation magnitudes of at least 3 out of 4 beams pass
        suspect: only two of the beams pass
        fail: one or none of the beams pass

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the TRDI ADCP data downloaded from
        OOINet in .netcdf format
    suspect: float
        The suspect threshold computed from the TRDI QA-QC Model
    fail: flot
        The fail threshold computed from the TRDI QA-QC Model

    Returns
    -------
    qc_flags: numpy.array[int]
        An array of QARTOD-style flags indicating the results of the QC
        test for each given datum
    """
    # Set up the qc_flags 
    qc_flags = np.ones(ds['correlation_magnitude_beam1'].shape, dtype=int)
    
    # Simplify implementation by adding booleans
    beam1_pass = (ds['correlation_magnitude_beam1'] > suspect).astype(int)
    beam2_pass = (ds['correlation_magnitude_beam2'] > suspect).astype(int)
    beam3_pass = (ds['correlation_magnitude_beam3'] > suspect).astype(int)
    beam4_pass = (ds['correlation_magnitude_beam4'] > suspect).astype(int)

    # Sum the results
    total_pass = beam1_pass + beam2_pass + beam3_pass + beam4_pass

    # Good values sum to 3 or 4
    # Suspect values sum to 2
    mask = (total_pass == 2)
    qc_flags[mask] = 3

    # Fail values sum to 0 or 1
    mask = (total_pass < 2)
    qc_flags[mask] = 4

    # Return the qc_flags
    return qc_flags


def percent_good_qc(ds: xarray.Dataset, suspect: float, fail: float) -> npt.NDArray[int]:
    """
    Determine ADCP QC based on the percent good returned for each
    beam and assign QARTOD-style flags. This algorithm uses thresholds
    computed using the TRDI ADCP Data AQ-QC Model rev12-1. The assigned 
    flag values are:

        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail. This algorithm

    Percent good is calculated from the best returns of either 3 and 4 beam
    solutions, since either may be used to calculate the seawater velocities.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the TRDI ADCP data downloaded from
        OOINet in .netcdf format
    suspect: float
        The suspect threshold computed from the TRDI QA-QC Model
    fail: flot
        The fail threshold computed from the TRDI QA-QC Model

    Returns
    -------
    qc_flags: numpy.array[int]
        An array of QARTOD-style flags indicating the results of the QC
        test for each given datum
    """
    # Merge the 3 and 4 beam solutions along a new axis and take the maximum
    percent_good = np.stack([ds['percent_good_3beam'].values, ds['percent_good_4beam'].values], axis=-1)
    percent_good = np.max(percent_good, axis=2)
    
    # Create a qc_flags array
    qc_flags = np.ones(percent_good.shape, dtype=int)
    
    # Find where the flags are suspect
    mask = (percent_good < suspect)
    qc_flags[mask] = 3

    # Find where the correlation magnitude is bad
    mask = (percent_good < fail)
    qc_flags[mask] = 4

    # Return the results
    return qc_flags


def merge_qc(test_results: list[npt.NDArray[int]]) -> npt.NDArray[int]:
    """
    Merge the results of the different QC tests into a single
    output. The results for the entire ensemble are:
    
        Pass:    100% of qc tests pass
        Suspect: At least 50% of tests pass or are suspect
        Fail:    Less than 50% of tests pass or are suspect

    The assigned QARTOD-style flag values are:
        1 = Pass
        3 = Suspect
        4 = Fail 

    Parameters
    ----------
    test_results: list[numpy.array[int]]
        A list containing all of the run individual ADCP
        QC test results

    Returns
    -------
    qc_flags: numpy.array[int]
        A numpy array that contains the combined results
        of the individual QC tests passed in test_results
    """
    n = len(test_results)
    qc_flags = np.zeros(test_results[0].shape, dtype=int)

    # First, calculate the most "inclusive" case of the number
    # of pass OR suspect tests and calculate the fraction 
    tests_suspect = np.zeros(test_results[0].shape, dtype=int)
    for test in test_results:
        suspect = ((test == 1) | (test == 3)).astype(int)
        tests_suspect = tests_suspect + suspect
    # Now calculate the fraction of tests that passed
    total_suspect = (tests_suspect / n)
    # Find where not enough tests pass and mark as fail
    mask = (total_suspect < 0.5)
    qc_flags[mask] = 4
    # Mark the rest as suspect. Will test for pass next
    mask = (total_suspect >= 0.5)
    qc_flags[mask] = 3

    # Now test for if all tests pass
    tests_passed = np.zeros(test_results[0].shape, dtype=int)
    # Sum up the number of "passes" in the qc_tests
    for test in test_results:
        passed = (test == 1).astype(int)
        tests_passed = tests_passed + passed
    # Now calculate the fraction of tests that passed
    total_passed = (tests_passed / n)
    mask = (total_passed == 1)
    qc_flags[mask] = 1

    # Return the result
    return qc_flags

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os

from ooi_data_explorations.common import inputs, m2m_collect, m2m_request, get_deployment_dates, \
    get_vocabulary, dt64_epoch, update_dataset, ENCODINGS


ADCP = {
    'bin_number': {
        'long_name': 'Bin Number',
        'comment': 'Number of the ADCP velocity bin. Number of bins is dependent on depth of deployment and frequency.',
        # 'units': '',    deliberately left blank, no units for this value
    },
    # PD0 fixed leader
    'frequency': {
        'long_name': 'Sysconfig Frequency',
        'comment': 'Workhorse transducer frequency',
        'units': 'kHz'
    },
    'vertical_orientation': {
        'long_name': 'Sysconfig Vertical Orientation',
        'comment': 'Whether vertical orientation is upward or downward',
        # 'units': '',    deliberately left blank, no units for this value
    },
    'bin_size': {
        'long_name': 'Bin Size',
        'comment': 'Contains the size of a bin in meters.',
        'units': 'm'
    },
    'bin_1_distance': {
        'long_name': 'Distance First Bin',
        'comment': 'Distance to the middle of the first depth bin.',
        'units': 'm'
    },
    'bin_depth': {
        'long_name': 'Bin Depth',
        'comment': ('Depth of each bin derived from the pressure record, the number of bins, the bin size, and the '
                    'distance to the first bin. Note, if the instrument is not equipped with a pressure sensor, then '
                    'either data from a co-located CTD or the estimated transducer depth (planned deployment depth) '
                    'is used to calculate the bin depth.'),
        'units': 'm'
    },

    # variable leader
    'ensemble_number': {
        'long_name': 'Ensemble Number',
        'comment': 'Sequential number of the ensemble to which the data applies',
        # 'units': '',    deliberately left blank, no units for this value
    },
    'speed_of_sound': {
        'comment': 'Contains either manual or calculated speed of sound',
        'long_name': 'Speed of Sound',
        'units': 'm s-1',
    },
    'transducer_depth': {
        'long_name': 'Estimated Transducer Depth',
        'comment': ('Estimated deployment depth of the ADCP, entered during configuration. Scaled to meters from '
                    'decimeters'),
        'units': 'm'
    },
    'heading': {
        'long_name': 'Heading',
        'comment': 'Measured heading of the ADCP, uncorrected for magnetic declination. Reported in decidegrees.',
        'units': 'degrees'
    },
    'pitch': {
        'long_name': 'Pitch',
        'comment': 'Measured pitch of the ADCP scaled to degrees from decidegrees.',
        'units': 'degrees'
    },
    'roll': {
        'long_name': 'Roll',
        'comment': 'Measured roll of the ADCP scaled to degrees from decidegrees.',
        'units': 'degrees'
    },
    'salinity': {
        'long_name': 'Transducer Salinity',
        'comment': ('Estimated salinity for the ADCP at the deployment site. Entered during configuration and used '
                    'to estimate the speed of sound.'),
        'units': '1'
    },
    'temperature': {
        'long_name': 'Sea Water Temperature',
        'standard_name': 'sea_water_temperature',
        'comment': ('In-situ sea water temperature measured at the transducer face, scaled to degrees_Celsius from'
                    'centidegree Celsius.'),
        'units': 'degrees_Celsius'
    },
    'heading_stdev': {
        'long_name': 'Heading Standard Deviation',
        'comment': 'Standard deviation of the heading reported in degrees',
        'units': 'degrees',
    },
    'pitch_stdev': {
        'long_name': 'Pitch Standard Deviation',
        'comment': 'Standard deviation of the pitch scaled to degrees from decidegrees',
        'units': 'degrees',
    },
    'roll_stdev': {
        'long_name': 'Roll Standard Deviation',
        'comment': 'Standard deviation of the roll scaled to degrees from decidegrees',
        'units': 'degrees',
    },
    'pressure': {
        'long_name': 'Pressure',
        'standard_name': 'sea_water_pressure_due_to_sea_water',
        'comment': ('ADCP pressure sensor value. Scaled to dbar from decaPascals. If the value is 0, the unit is not '
                    'equipped with a pressure sensor.'),
        'units': 'dbar'
    },
    'pressure_variance': {
        'long_name': 'Pressure Variance',
        'comment': 'Variability in the pressure reading during the ensemble averaging period.',
        'units': 'dbar'
    },

    # velocity packets
    'eastward_seawater_velocity': {
        'long_name': 'Eastward Seawater Velocity',
        'comment': ('A velocity profile includes water velocity (speed & direction) throughout the depth range of an '
                    'ADCP sensor. This instance is the eastward seawater velocity component corrected for magnetic '
                    'declination and scaled to m/s.'),
        'standard_name': 'eastward_sea_water_velocity',
        'data_product_identifier': 'VELPROF-VLE_L1',
        'units': 'm s-1',
        '_FillValue': np.nan
    },
    'northward_seawater_velocity': {
        'long_name': 'Northward Seawater Velocity',
        'comment': ('A velocity profile includes water velocity (speed & direction) throughout the depth range of an '
                    'ADCP sensor. This instance is the eastward seawater velocity component corrected for magnetic '
                    'declination and scaled to m/s.'),
        'standard_name': 'northward_sea_water_velocity',
        'data_product_identifier': 'VELPROF-VLN_L1',
        'units': 'm s-1',
        '_FillValue': np.nan
    },
    'vertical_seawater_velocity': {
        'long_name': 'Estimated Vertical Seawater Velocity',
        'comment': ('A velocity profile includes water velocity (speed & direction) throughout the depth range of an '
                    'ADCP sensor. This instance is the vertical seawater velocity component scaled to m/s.'),
        'standard_name': 'upward_sea_water_velocity',
        'data_product_identifier': 'VELPROF-VLU_L1',
        'units': 'm s-1',
        '_FillValue': np.nan
    },
    'error_velocity': {
        'long_name': 'Error Velocity',
        'comment': ('A velocity profile includes water velocity (speed & direction) throughout the depth range of an '
                    'ADCP sensor. This instance is the error velocity scaled to m/s..'),
        'data_product_identifier': 'VELPROF-EVL_L1',
        'units': 'm s-1',
        '_FillValue': np.nan
    },

    # correlation magnitudes
    'correlation_magnitude_beam1': {
        'long_name': 'Correlation Magnitude Beam 1',
        'comment': ('Magnitude of the normalized echo auto-correlation at the lag used for estimating the Doppler '
                    'phase change. 0 represents no correlation and 255 represents perfect correlation.'),
        'units': 'counts'
    },
    'correlation_magnitude_beam2': {
        'long_name': 'Correlation Magnitude Beam 2',
        'comment': ('Magnitude of the normalized echo auto-correlation at the lag used for estimating the Doppler '
                    'phase change. 0 represents no correlation and 255 represents perfect correlation.'),
        'units': 'counts'
    },
    'correlation_magnitude_beam3': {
        'long_name': 'Correlation Magnitude Beam 3',
        'comment': ('Magnitude of the normalized echo auto-correlation at the lag used for estimating the Doppler '
                    'phase change. 0 represents no correlation and 255 represents perfect correlation.'),
        'units': 'counts'
    },
    'correlation_magnitude_beam4': {
        'long_name': 'Correlation Magnitude Beam 4',
        'comment': ('Magnitude of the normalized echo auto-correlation at the lag used for estimating the Doppler '
                    'phase change. 0 represents no correlation and 255 represents perfect correlation.'),
        'units': 'counts'
    },

    # echo intensities
    'echo_intensity_beam1': {
        'long_name': 'Echo Intensity Beam 1',
        'comment': ('Echo Intensity is the acoustic return signal per beam that is output directly from the ADCP. '
                    'This is the raw measurement used to calculate the echo intensity data product for the beam.'),
        'data_product_identifier': 'ECHOINT-B1_L0',
        'units': 'counts'
    },
    'echo_intensity_beam2': {
        'long_name': 'Echo Intensity Beam 2',
        'comment': ('Echo Intensity is the acoustic return signal per beam that is output directly from the ADCP. '
                    'This is the raw measurement used to calculate the echo intensity data product for the beam.'),
        'data_product_identifier': 'ECHOINT-B2_L0',
        'units': 'counts'
    },
    'echo_intensity_beam3': {
        'long_name': 'Echo Intensity Beam 3',
        'comment': ('Echo Intensity is the acoustic return signal per beam that is output directly from the ADCP. '
                    'This is the raw measurement used to calculate the echo intensity data product for the beam.'),
        'data_product_identifier': 'ECHOINT-B3_L0',
        'units': 'counts'
    },
    'echo_intensity_beam4': {
        'long_name': 'Echo Intensity Beam 4',
        'comment': ('Echo Intensity is the acoustic return signal per beam that is output directly from the ADCP. '
                    'This is the raw measurement used to calculate the echo intensity data product for the beam.'),
        'data_product_identifier': 'ECHOINT-B4_L0',
        'units': 'counts'
    },

    # percent good
    'percent_good_3beam': {
        'long_name': 'Percent Good 3 Beams',
        'comment': ('Percentage of velocity data collected in an ensemble average that were calculated with just 3 '
                    'beams.'),
        'units': 'percent'
    },
    'percent_transforms_reject': {
        'long_name': 'Percent Transforms Rejected',
        'comment': ('Percentage of transformations rejected in an ensemble average (error velocity that was higher '
                    'than the WE-command setting)'),
        'units': 'percent'
    },
    'percent_bad_beams': {
        'long_name': 'Percent Bad Beams',
        'comment': ('Percentage of velocity data collected in an ensemble average that were rejected because not '
                    'enough beams had good data.'),
        'units': 'percent'
    },
    'percent_good_4beam': {
        'long_name': 'Percent Good 4 Beams',
        'comment': 'Percentage of velocity data collected in an ensemble average that were calculated with all 4 beams',
        'units': 'percent'
    }
}


def adcp_instrument(earth, engineering):
    """
    Takes adcp data recorded either by the instrument or data loggers used in
    the CGSN/EA moorings and cleans up the data set to make it more user-
    friendly. The resulting data set represents a subset of all the variables
    available in full PD0 record.

    The full PD0 record is available from the OOI system, though it was split
    into two separate data sets before it was ingested into the system. Data
    from both data sets are needed to properly process the data; particularly
    with regard to QC assessments. This module pulls the needed parameters
    out of both sources, re-combines them into a single data set and then
    cleans up the data set for easier use.

    :param earth: initial PD0 formatted ADCP data set downloaded from OOI via
        the M2M system (represents a subset of the full PD0 formatted data set)
    :param engineering: remaining PD0 formatted ADCP data set downloaded from
        OOI via the M2M system (represents the remaining subset of the full
        PD0 formatted data set)
    :return adcp: cleaned up data set
    """
    # drop select variables from the earth data set:
    earth = earth.reset_coords()
    drop_vars = ['corrected_echo_intensity_beam1', 'corrected_echo_intensity_beam2', 'corrected_echo_intensity_beam3',
                 'corrected_echo_intensity_beam4', 'water_velocity_up', 'error_seawater_velocity']
    for v in drop_vars:
        if v in earth.variables:
            earth = earth.drop(v)

    # convert the time values from a datetime64[ns] object to a floating point number with the time in seconds
    earth['internal_timestamp'] = ('time', dt64_epoch(earth.internal_timestamp))
    earth['internal_timestamp'].attrs = dict({
        'long_name': 'Internal ADCP Clock Time',
        'standard_name': 'time',
        'units': 'seconds since 1970-01-01T00:00:00.000Z',
        'calendar': 'gregorian',
        'comment': ('Comparing the instrument internal clock versus the GPS referenced sampling time will allow for '
                    'calculations of the instrument clock offset and drift. Useful when working with the '
                    'recovered instrument data where no external GPS referenced clock is available.')
    })
