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
