#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import xarray as xr

# constants used by the ASIMET SWND v4.11 firmware processing
PI2 = 6.283185       # 2 * pi
DEG2RAD = 0.0174533  # convert to radians (deg * (2 * pi / 360))
RAD2DEG = 57.29578   # convert to degrees (rad * (360 / 2 * pi))


def wind_binning(tbin):
    """
    Function to bin the 5-second wind data into 1-minute averages. The wind
    components are then re-calculated using the scalar average wind speed and
    vector averaged wind direction.

    :param tbin: grouped pandas dataframe with ~1-minute of 5-second wind data
    :return avg: dataframe with the averaged wind data
    """
    # calculate the 1-minute simple averages (using a median) for the bin
    avg = tbin.median(dim='time')

    # add the min and max wind speeds for the bin
    avg['wind_speed_min'] = tbin['wind_speed'].min(dim='time')
    avg['wind_speed_max'] = tbin['wind_speed'].max(dim='time')

    # convert the averaged compass heading back to degrees
    avg['heading'] = avg['heading'] * RAD2DEG

    # use the scalar-averaged wind speed and the wind direction calculated from vector averages to re-calculate the
    # eastward and northward wind components per directions provided by the NDBC (https://www.ndbc.noaa.gov/wndav.shtml)
    wind_direction = np.arctan2(avg['eastward_wind'], avg['northward_wind'])  # reported in radians
    wind_direction = np.where(wind_direction < 0, wind_direction + PI2, wind_direction)  # 0 to 360 degrees, in radians
    avg['eastward_wind'] = avg['wind_speed'] * np.sin(wind_direction)
    avg['northward_wind'] = avg['wind_speed'] * np.cos(wind_direction)

    # replace the averaged wind direction with one derived from the vector averages (converting radians to degrees)
    avg['wind_direction'] = wind_direction * RAD2DEG  # convert to degrees (rad * (360 / 2 * pi))

    # return the resulting 1-minute averaged bin
    return avg


def proc_wind(raw):
    """
    Main ASIMET Sonic Wind (SWND) module processing function. Loads the raw
    5-second parsed data and calculates different wind products following the
    NDBC convention. For more information on the ASIMET system and the SWND
    module, see the ASIMET website:

        "https://www.whoi.edu/what-we-do/explore/instruments/\
            instruments-sensors-samplers/\
            air-sea-interaction-meteorology-the-asimet-system/"

    :param raw: xarray dataset with the raw, 5-second wind data
    :return swnd: xarray dataset with the processed SWND data
    """
    # convert the Gill wind components to the oceanographic convention (relative to the east and north axis
    # of the instrument, rather than the U and V axis of the instrument).
    raw['eastward_wind_relative'] = -1 * raw['v_axis_wind_speed']  # rename and convert v-axis to positive eastward
    raw['northward_wind_relative'] = raw['u_axis_wind_speed']      # rename u-axis to northward

    # drop the original wind components and the DCL date and time string
    raw = raw.drop(columns=['u_axis_wind_speed', 'v_axis_wind_speed', 'dcl_date_time_string'])

    # convert the compass heading to radians
    raw['heading'] = raw['heading'] * DEG2RAD  # convert to radians (deg * (2 * pi / 360))

    # calculate the wind speed
    raw['wind_speed'] = np.sqrt(raw['eastward_wind_relative']**2 + raw['northward_wind_relative']**2)

    # calculate the relative wind direction, first correcting the northward wind component to avoid
    # divide by zero errors
    raw['northward_wind_relative'] = np.where(raw['northward_wind_relative'] == 0, 0.00001,
                                               raw['northward_wind_relative'])
    direction = np.arctan2(raw['eastward_wind_relative'], raw['northward_wind_relative'])

    # convert to 0 to 360 degrees, but leave in radians for further calculations
    direction = np.where(direction < 0, direction + PI2, direction)

    # if the wind speed is less than 0.05 m/s, forward fill with the last valid value (per the Gill WindMasterII manual)
    direction = np.where(raw['wind_speed'] < 0.05, np.roll(direction, 1), direction)

    # calculate the wind direction relative to magnetic north (using the compass heading and relative wind direction)
    wind_direction = direction + raw['heading']
    wind_direction = np.where(wind_direction > PI2, wind_direction - PI2, wind_direction)

    # now compute the eastward and northward wind components relative to magentic north
    raw['eastward_wind'] = raw['wind_speed'] * np.sin(wind_direction)
    raw['northward_wind'] = raw['wind_speed'] * np.cos(wind_direction)

    # create an xarray data set from the data frame
    raw = xr.Dataset.from_dataframe(raw)

    # shift the time so subsequent resampling bins center the data in the middle of the 1-minute bin
    raw['time'] = raw.time + np.timedelta64(30, 's')

    # resample the data to 1-minute bins using the wind_binning function defined above
    swnd = raw.resample(time='1Min', skipna=True).map(wind_binning)

    # return the final dataset
    return swnd
