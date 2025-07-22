#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gsw
import ppigrf
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
import xarray as xr

from ooi_data_explorations.common import inputs, m2m_collect, m2m_request, get_deployment_dates, \
    get_vocabulary, dt64_epoch, update_dataset, ENCODINGS
from pyseas.data.adcp_functions import magnetic_correction

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
    'heading': {
        'long_name': 'Heading',
        'comment': ('Measured heading of the ADCP, uncorrected for magnetic declination scaled to degrees from '
                    'centidegrees.'),
        'units': 'degrees'
    },
    'pitch': {
        'long_name': 'Pitch',
        'comment': 'Measured pitch of the ADCP scaled to degrees from centidegrees.',
        'units': 'degrees'
    },
    'roll': {
        'long_name': 'Roll',
        'comment': 'Measured roll of the ADCP scaled to degrees from centidegrees.',
        'units': 'degrees'
    },
    'salinity': {
        'long_name': 'Estimated Transducer Salinity',
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
    'pressure': {
        'long_name': 'Pressure',
        'standard_name': 'sea_water_pressure_due_to_sea_water',
        'comment': ('ADCP pressure sensor value. Scaled to dbar from decaPascals. If the value is 0, the unit is not '
                    'equipped with a pressure sensor.'),
        '_FillValue': 0,
        'units': 'dbar'
    },
    # velocity packets
    'eastward_seawater_velocity_est': {
        'long_name': 'Estimated Eastward Seawater Velocity',
        'comment': ('A velocity profile includes water velocity (speed & direction) throughout the depth range of an '
                    'ADCP sensor. This instance is the eastward seawater velocity component uncorrected for magnetic '
                    'declination as reported by the instrument. Considered an estimate of the true eastward velocity '
                    'component as it is uncorrected for magnetic declination.'),
        'data_product_identifier': 'VELPROF-VLE_L0',
        'units': 'mm s-1',
        '_FillValue': -32768
    },
    'northward_seawater_velocity_est': {
        'long_name': 'Estimated Northward Seawater Velocity',
        'comment': ('A velocity profile includes water velocity (speed & direction) throughout the depth range of an '
                    'ADCP sensor. This instance is the northward seawater velocity component uncorrected for magnetic '
                    'declination as reported by the instrument. Considered an estimate of the true northward velocity '
                    'component as it is uncorrected for magnetic declination.'),
        'data_product_identifier': 'VELPROF-VLN_L0',
        'units': 'mm s-1',
        '_FillValue': -32768
    },
    'vertical_seawater_velocity': {
        'long_name': 'Vertical Seawater Velocity',
        'comment': ('A velocity profile includes water velocity (speed & direction) throughout the depth range of an '
                    'ADCP sensor. This instance is the vertical seawater velocity component as reported by the '
                    'instrument'),
        'standard_name': 'upward_sea_water_velocity',
        'data_product_identifier': 'VELPROF-VLU_L0',
        'units': 'mm s-1',
        '_FillValue': -32768
    },
    'error_velocity': {
        'long_name': 'Error Velocity',
        'comment': ('A velocity profile includes water velocity (speed & direction) throughout the depth range of an '
                    'ADCP sensor. This instance is the error velocity component as reported by the instrument.'),
        'data_product_identifier': 'VELPROF-EVL_L0',
        'units': 'mm s-1',
        '_FillValue': -32768
    },
    'eastward_seawater_velocity': {
        'long_name': 'Eastward Seawater Velocity',
        'comment': ('Eastward sea water velocity component in Earth coordinates corrected for magnetic declination ' +
                    'and scaled to standard units of m s-1.'),
        'standard_name': 'eastward_sea_water_velocity',
        'data_product_identifier': 'VELPROF-VLE_L1',
        'ancillary_variables': 'eastward_seawater_velocity_est, northward_seawater_velocity_est, time, lat, lon, z',
        'units': 'm s-1',
        '_FillValue': np.nan
    },
    'northward_seawater_velocity': {
        'long_name': 'Northward Seawater Velocity',
        'comment': ('Northward sea water velocity component in Earth coordinates corrected for magnetic declination ' +
                    'and scaled to standard units of m s-1.'),
        'standard_name': 'northward_sea_water_velocity',
        'data_product_identifier': 'VELPROF-VLN_L1',
        'ancillary_variables': 'eastward_seawater_velocity_est, northward_seawater_velocity_est, time, lat, lon, z',
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
    'backscatter_beam1': {
        'long_name': 'Estimated Acoustic Backscatter Beam 1',
        'comment': ('Acoustic backscatter is the strength of the returned sound wave pulse transmitted by the ADCP. ' +
                    'Acoustic backscatter can be used as an indicator of the amount of sediment or organisms in the ' +
                    'water column, as well as the quality of a velocity measurement. It is estimated from the echo ' +
                    'intensity measurement using default conversion factors provided in the vendor documentation.'),
        'data_product_identifier': 'ECHOINT-B1_L1',
        'ancillary_variables': 'echo_intensity_beam1',
        'units': 'dB'
    },
    'backscatter_beam2': {
        'long_name': 'Estimated Acoustic Backscatter Beam 2',
        'comment': ('Acoustic backscatter is the strength of the returned sound wave pulse transmitted by the ADCP. ' +
                    'Acoustic backscatter can be used as an indicator of the amount of sediment or organisms in the ' +
                    'water column, as well as the quality of a velocity measurement. It is estimated from the echo ' +
                    'intensity measurement using default conversion factors provided in the vendor documentation.'),
        'data_product_identifier': 'ECHOINT-B2_L1',
        'ancillary_variables': 'echo_intensity_beam2',
        'units': 'dB'
    },
    'backscatter_beam3': {
        'long_name': 'Estimated Acoustic Backscatter Beam 3',
        'comment': ('Acoustic backscatter is the strength of the returned sound wave pulse transmitted by the ADCP. ' +
                    'Acoustic backscatter can be used as an indicator of the amount of sediment or organisms in the ' +
                    'water column, as well as the quality of a velocity measurement. It is estimated from the echo ' +
                    'intensity measurement using default conversion factors provided in the vendor documentation.'),
        'data_product_identifier': 'ECHOINT-B3_L1',
        'ancillary_variables': 'echo_intensity_beam3',
        'units': 'dB'
    },
    'backscatter_beam4': {
        'long_name': 'Estimated Acoustic Backscatter Beam 4',
        'comment': ('Acoustic backscatter is the strength of the returned sound wave pulse transmitted by the ADCP. ' +
                    'Acoustic backscatter can be used as an indicator of the amount of sediment or organisms in the ' +
                    'water column, as well as the quality of a velocity measurement. It is estimated from the echo ' +
                    'intensity measurement using default conversion factors provided in the vendor documentation.'),
        'data_product_identifier': 'ECHOINT-B4_L1',
        'ancillary_variables': 'echo_intensity_beam4',
        'units': 'dB'
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

# Can use these based on NDBC but recommend using the TRDI QA/QC Model Rev12-1 (no references available for
# these values because that's how Andrew Reed works. Sloppy, careless, and unprofessional.)
QCThresholds = {
    'error_velocity': {
        'pass': 0.05,
        'suspect': 0.10,  # Oh, I'm Andrew Reed, I don't provide suspect thresholds but my code requires it...
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

    The sidelobe interference depth is calculated following 
    Lentz et al. (2022) where:
    
        z_ic = ha * [1 - cos(theta)] + 3 * delta_Z / 2
        
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
    depth = ds['depth']
    ha = depth.interpolate_na(dim='time', method='linear')

    # Next, get the beam angle
    theta = np.deg2rad(theta)

    # Grab the cell length
    delta_z = ds['cell_length'].mean(skipna=True)

    # Calculate the range of cells contaminated by sidelobe interference
    z_ic = ha * (1 - np.cos(theta)) + 3 * delta_z / 2
    return z_ic


def sidelobe_qc(ds: xr.Dataset) -> xr.Dataset:
    """
    Add sidelobe interference QARTOD-style flags to the ADCP dataset

    Assignment of QARTOD style quality flags to the ADCP velocity data based
    on estimation of sidelobe contamination. The sidelobe interference depth 
    is calculated following Lentz et al. (2022) where:
    
        z_ic = ha * [1 - cos(theta)] + 3 * delta_Z / 2
        
    z_ic is the depth above which there is sidelobe interference,
    ha is the transducer face depth, theta is the beam angle, and
    delta_Z is the cell-bin depth. Bin depths less than z_ic are
    considered contaminated and the data is considered bad. The assigned flag
    values are:
    
        1 = Pass
        4 = Fail

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the TRDI ADCP data downloaded from
        OOINet in NetCDF format

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
        'long_name': 'Sidelobe Contamination QC Summary Flag',
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
    Determine ADCP QC based on Error velocity and assign QARTOD-style flags.
    This algorithm uses thresholds computed using the TRDI ADCP Data QA-QC
    Model rev12-1. The assigned flag values are:

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
        OOINet in NetCDF format
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
    using the TRDI ADCP Data QA-QC Model rev12-1. The assigned 
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
        OOINet in NetCDF format
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
    Determine ADCP QC based on the horizontal velocity and assign
    QARTOD-style flags. This algorithm uses thresholds computed
    using the TRDI ADCP Data QA-QC Model rev12-1. The assigned 
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
        OOINet in NetCDF format
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
    using the TRDI ADCP Data QA-QC Model rev12-1. The assigned
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
        OOINet in NetCDF format
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
    beam1_pass = int(ds['correlation_magnitude_beam1'] > suspect)
    beam2_pass = int(ds['correlation_magnitude_beam2'] > suspect)
    beam3_pass = int(ds['correlation_magnitude_beam3'] > suspect)
    beam4_pass = int(ds['correlation_magnitude_beam4'] > suspect)

    # Sum the results
    total_pass = beam1_pass + beam2_pass + beam3_pass + beam4_pass

    # Good values sum to 3 or 4, suspect values sum to 2
    mask = (total_pass == 2)
    qc_flags[mask] = 3

    # Fail values sum to 0 or 1
    mask = (total_pass < 2)
    qc_flags[mask] = 4

    # Return the qc_flags
    return qc_flags


def percent_good_qc(ds: xr.Dataset, suspect: float, fail: float) -> npt.NDArray[int]:
    """
    Determine ADCP QC based on the percent good returned for each
    beam and assign QARTOD-style flags. This algorithm uses thresholds
    computed using the TRDI ADCP Data QA-QC Model rev12-1. The assigned 
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
        OOINet in NetCDF format
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
        A list containing all the individual ADCP QC test results

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
        suspect = int((test == 1) | (test == 3))
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
        passed = int(test == 1)
        tests_passed = tests_passed + passed

    # Now calculate the fraction of tests that passed
    total_passed = (tests_passed / n)
    mask = (total_passed == 1)
    qc_flags[mask] = 1

    # Return the result
    return qc_flags


def adcp_instrument(ds: xr.Dataset) -> xr.Dataset:
    """
    Takes adcp data recorded either by the instrument or data loggers used in
    the CGSN/EA moorings and cleans up the data set to make it more
    user-friendly. Additionally, a suite of QC checks are added to help assess
    the quality of the data. The resulting data set represents a subset of all
    the variables available in a full PD0 record.

    :param ds: initial PD0 formatted ADCP data set downloaded from OOI via
        the M2M system (represents a subset of the full PD0 formatted data set)
    :return adcp: cleaned up data set with QC assessments added
    """
    # drop select variables from the data set:
    ds = ds.reset_coords()
    drop_vars = ['corrected_echo_intensity_beam1', 'corrected_echo_intensity_beam2', 
                 'corrected_echo_intensity_beam3', 'corrected_echo_intensity_beam4',
                 'ctdbp_cdef_dcl_instrument-depth', 'non_zero_pressure', 'non_zero_depth',
                 'depth_from_pressure']
    for v in drop_vars:
        if v in ds.variables:
            ds = ds.drop_vars(v)

    # rename selected variables
    rename = {
        'bin': 'bin_number',
        'sysconfig_vertical_orientation': 'vertical_orientation',
        'transducer_depth': 'transducer_pressure',
        'int_ctd_pressure': 'ctd_pressure',
    }
    for key, value in rename.items():
        if key in ds.variables:
            ds = ds.rename({key: value})
            ds[value].attrs['ooinet_variable_name'] = key

    # convert some of the variables to floating point numbers (vendor stores some values as integers)
    ds['heading'] = ds['heading'] / 100.0  # convert from centidegrees to degrees
    ds['pitch'] = ds['pitch'] / 100.0  # convert from centidegrees to degrees
    ds['roll'] = ds['roll'] / 100.0  # convert from centidegrees to degrees
    ds['bin_1_distance'] = ds['bin_1_distance'] / 100.0  # convert from cm to m
    ds['cell_length'] = ds['cell_length'] / 100.0  # convert from cm to m

    # Clean-up the mess that is the depth and pressure variables. Either the ADCP is equipped with a pressure
    # sensor or it is not. If it is, use that record to calculate the depth of the ADCP. If it is not, use the
    # data from a co-located CTD or the estimated transducer depth (planned deployment depth if the CTD is unavailable)
    # to calculate the depth of the ADCP. This depth record is then used to calculate the bin depths. The current
    # method employed by OOI results in a depth record that can have an offset from the actual deployment depth
    # when compared to the CTD or ADCP pressure records.
    if np.all(ds['transducer_pressure'].values == 0):
        # if all the transducer pressure values are 0, the ADCP is not equipped with a pressure sensor
        ds['transducer_pressure'] = ds['transducer_pressure'].astype(float) * np.nan  # convert record to NaNs
        if 'ctd_pressure' in ds.variables:  # CTD pressure is available, use that to calculate the depth
            offset = np.where('RID' in ds.attrs['node'], 1.0, 0.0)  # depth offset for ADCPs on NSIFs
            ds['depth'] = gsw.z_from_p(ds['ctd_pressure'], ds.attrs['latitude']) + offset
        else:  # no CTD pressure is available, use the deployment depth to estimate the depth
            ds['ctd_pressure'] = ds['transducer_pressure'] * np.nan  # set CTD pressure to NaN
            vocab = get_vocabulary(ds.attrs['subsite'], ds.attrs['node'], ds.attrs['sensor'])[0]
            ds['depth'] = vocab['maxdepth']

        # since the CTD can fail mid-deployment, we can end up with cases where the pressure record is NaN-filled
        ds['depth'] = ds['depth'].fillna(ds['depth'].mean())  # fill NaNs with the mean depth
    else:
        ds['transducer_pressure'] = ds['transducer_pressure'] / 10.0  # convert from decaPascals to dbar
        ds['depth'] = -1 * gsw.z_from_p(ds['transducer_pressure'], ds.attrs['lat'])  # convert pressure to depth

    # calculate the bin depths
    z_sign = np.atleast_2d(np.where(ds['vertical_orientation'].values == 0, 1.0, -1.0)).T
    ds['bin_depths'] = ds['depth'] + z_sign * (ds['bin_1_distance'] + ds['cell_length'] * ds['bin_number'])

    # convert the time values from a datetime64[ns] object to a floating point number with the time in seconds
    ds['internal_timestamp'] = ('time', dt64_epoch(ds.internal_timestamp))
    ds['internal_timestamp'].attrs = dict({
        'long_name': 'Internal ADCP Clock Time',
        'standard_name': 'time',
        'units': 'seconds since 1970-01-01T00:00:00.000Z',
        'calendar': 'gregorian',
        'comment': ('Comparing the instrument internal clock versus the GPS referenced sampling time will allow for '
                    'calculations of the instrument clock offset and drift. Useful when working with the '
                    'recovered instrument data where no external GPS referenced clock is available.')
    })

    # OOI uses an old version (2015) of the WMM (World Magnetic Model) to calculate the magnetic declination. In
    # addition to being outdated, the WMM in use is only valid for a limited time period, covering data collected
    # between 2015-01-01 and 2019-12-31. The current International Geomagnetic Reference Field (IGRF, Version 14)
    # model is valid from 1900-01-01 to 2029-12-31. We can use the IGRF model to calculate the correct magnetic
    # declination for all sites and deployments, regardless of the date of the data, and from there properly
    # calculate the eastward and northward seawater velocity components adjusted for magnetic declination. Granted,
    # the total error is small, but it is still an error that should and can be corrected.
    Be, Bn, Bu = ppigrf.igrf(ds.attrs['lon'], ds.attrs['lat'], 0, ds['time'])  # returns east, north, up
    incln, decln = ppigrf.get_inclination_declination(Be, Bn, Bu, degrees=True)
    east, north = magnetic_correction(decln, ds['water_velocity_east'], ds['water_velocity_north'])
    ds['eastward_seawater_velocity'] = (('time', 'bin_number'), east / 1000.0)  # convert from mm/s to m/s
    ds['northward_seawater_velocity'] = (('time', 'bin_number'), north / 1000.0)  # convert from mm/s to m/s
    # The vertical seawater velocity is the vertical component of the water velocity, which is not corrected for
    # magnetic declination, but is scaled to m/s
    ds['upward_seawater_velocity'] = ds['water_velocity_up'] / 1000.0  # convert from mm/s to m/s
    # The error velocity is the difference between the measured velocity and the expected velocity, which is not
    # corrected for magnetic declination, but is scaled to m/s
    ds['error_seawater_velocity'] = ds['error_velocity'] / 1000.0  # convert from mm/s to m/s

    # now add the QC assessments to the dataset
    error_velocity = error_velocity_qc(ds, QCThresholds['error_velocity']['fail'])
    vertical_velocity = vertical_velocity_qc(ds, QCThresholds['upward_seawater_velocity']['suspect'], QCThresholds['vertical_velocity']['fail'])
    horizontal_speed = horizontal_speed_qc(ds, QCThresholds['horizontal_speed']['suspect'], QCThresholds['horizontal_speed']['fail'])
    correlation_magnitude = correlation_magnitude_qc(ds, QCThresholds['correlation_magnitude']['suspect'], QCThresholds['correlation_magnitude']['fail'])
    percent_good = percent_good_qc(ds, QCThresholds['percent_good']['ADCPT']['suspect'], QCThresholds['percent_good']['ADCPT']['fail'])
    
    # merge the QC results into a single summary flag
    qc_results = [error_velocity, vertical_velocity, horizontal_speed, correlation_magnitude, percent_good]
    ds['adcp_qc_summary_flag'] = ('time', merge_qc(qc_results))

    # set the attributes for the new QC variable
    ds['adcp_qc_summary_flag'].attrs = dict({
        'long_name': 'ADCP QC Summary Flag',
        'standard_name': 'quality_flag',
        'comment': ('A QARTOD style summary flag indicating the quality of the ADCP data, where the values are '
                    '1 == pass, 2 == not evaluated, 3 == suspect or of high interest, 4 == fail, and 9 == missing.'),
        'flag_values': np.array([1, 2, 3, 4, 9]),
        'flag_meanings': 'pass not_evaluated suspect_or_of_high_interest fail missing'
    })

    # return the cleaned up dataset
    return adcp
