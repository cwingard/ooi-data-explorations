#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import re

from ooi_data_explorations.common import inputs, m2m_collect, m2m_request, get_deployment_dates, \
    get_vocabulary, dt64_epoch, update_dataset, ENCODINGS


def pco2a_datalogger(ds, burst=False):
    """
    Takes PCO2A data recorded by the data loggers used in the CGSN/EA moorings
    and cleans up the data set to make it more user-friendly. Primary task is
    renaming parameters and dropping some that are of limited use. Additionally,
    re-organize some of the variables to permit better assessments of the data.

    :param ds: initial PCO2A data set downloaded from OOI via the M2M system
    :param burst: resample the data to an hourly, burst averaged time interval
    :return ds: cleaned up data set
    """
    # drop some of the variables:
    #   ### OOI generated parameters
    #   date_time_string == internal_timestamp, redundant so can remove
    #   dcl_controller_timestamp == time, redundant so can remove
    #   supply_voltage == not used
    #   ### Data products from upstream processing used to calculate the normalized 10 m wind, but are not needed
    #   eastward_velocity
    #   northward_velocity
    #   air_temperature
    #   met_relwind_speed
    #   longwave_irradiance
    #   shortwave_irradiance
    #   relative_humidity
    #   barometric_pressure
    #   precipitation
    shared = ['date_time_string', 'dcl_controller_timestamp']
    if 'supply_voltage' in ds.variables:
        # ... because it is in the telemetered, but not the recovered_host ...
        shared.append('supply_voltage')

    ds = ds.drop_vars(shared)

    # get rid of the older QC tests (OBE with the newer QARTOD tests) and the QARTOD executed test strings
    for v in ds.variables:
        if any(qc in v for qc in ['qc_executed', 'qc_results', 'qartod_executed']):
            ds = ds.drop_vars(v)

    # determine if the upstream parameters are present. delete them if needed, otherwise add the required ones to make
    # sure the NetCDF files are consistent
    upstream = ['eastward_velocity', 'northward_velocity', 'air_temperature', 'met_relwind_speed',
                'longwave_irradiance', 'shortwave_irradiance', 'relative_humidity', 'barometric_pressure',
                'precipitation']
    if 'eastward_velocity' in ds.variables:
        ds = ds.drop_vars(upstream)
    else:
        # METBK data was missing, add variables below to keep data sets consistent
        ds['sea_surface_temperature'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_surface_temperature'].attrs = {
            'long_name': 'Sea Surface Temperature',
            'standard_name': 'sea_surface_temperature',
            'comment': ('Normally this would be sea surface temperature data from a co-located CTD. However, data ' +
                        'from that sensor is unavailable. This value has been filled with NaNs to preserve the ' +
                        'structure of the data set.'),
            'units': 'degree_Celsius',
            'data_product_identifier': 'TEMPSRF_L1',
            'instrument': (ds.attrs['subsite'] + '-SBD11-06-METBKA000'),
            'stream': 'metbk_a_dcl_instrument'
        }

        ds['met_salsurf'] = ('time', ds['deployment'].data * np.nan)
        ds['met_salsurf'].attrs = {
            'long_name': 'Sea Surface Practical Salinity',
            'standard_name': 'sea_surface_salinity',
            'units': '1e-3',
            'comment': ('Normally this would be sea surface salinity data from a co-located CTD. However, data from ' +
                        'that sensor is unavailable. This value has been filled with NaNs to preserve the structure ' +
                        'of the data set.'),
            'data_product_identifier': 'SALSURF_L2',
            'instrument': (ds.attrs['subsite'] + '-SBD11-06-METBKA000'),
            'stream': 'metbk_a_dcl_instrument'
        }

        ds['met_wind10m'] = ('time', ds['deployment'].data * np.nan)
        ds['met_wind10m'].attrs = {
            'long_name': 'Normalized Wind Speed at 10 m',
            'standard_name': 'wind_speed',
            'units': 'm s-1',
            'comment': ('Normally this would be the modelled wind speed at a reference height of 10 m from a ' +
                        'co-located wind sensor. However, data from that sensor is unavailable. This value has been ' +
                        'filled with NaNs to preserve the structure of the data set.'),
            'data_product_identifier': 'WIND10M_L2',
            'instrument': (ds.attrs['subsite'] + '-SBD11-06-METBKA000'),
            'stream': 'metbk_hourly'
        }

    # convert the internal timestamp values from a datetime64[ns] object to a floating point number, time in seconds
    ds['internal_timestamp'] = ('time', dt64_epoch(ds.internal_timestamp))
    ds['internal_timestamp'].attrs = dict({
        'long_name': 'Internal pCO2-Pro Clock Time',
        'standard_name': 'time',
        'units': 'seconds since 1970-01-01 00:00:00 0:00',
        'calendar': 'gregorian',
        'comment': ('Comparing the instrument internal clock versus the GPS referenced sampling time will allow for ' +
                    'calculations of the instrument clock offset and drift.')
    })

    # rename variables to get a cleaner set variables and attributes
    rename = {
        'met_salsurf': 'sea_surface_salinity',
        'met_wind10m': 'normalized_10m_wind',
        'pco2_co2flux': 'sea_air_co2_flux',
    }
    ds = ds.rename(rename)
    for key, value in rename.items():   # bulk attribute update...
        ds[value].attrs['ooinet_variable_name'] = key

    ds['sea_air_co2_flux'].attrs['ancillary_variables'] = ('partial_pressure_co2_atm partial_pressure_co2_ssw ' +
                                                           'sea_surface_temperature sea_surface_salinity ' +
                                                           'normalized_10m_wind')

    # reset incorrectly formatted temperature units
    temp_vars = ['sea_surface_temperature', 'avg_irga_temperature', 'humidity_temperature', 'irga_detector_temperature',
                 'irga_source_temperature']
    for var in temp_vars:
        ds[var].attrs['units'] = 'degree_Celsius'

    # reset incorrectly set attributes for salinity and wind speed
    ds['sea_surface_salinity'].attrs['standard_name'] = 'sea_surface_salinity'
    ds['sea_surface_salinity'].attrs['long_name'] = 'Sea Surface Practical Salinity'
    ds['sea_surface_salinity'].attrs['units'] = '1e-3'
    ds['normalized_10m_wind'].attrs['standard_name'] = 'wind_speed'
    ds['normalized_10m_wind'].attrs['long_name'] = 'Normalized Wind Speed at 10 m'

    if burst:   # re-sample the data to a defined time interval using a median average
        burst = ds.copy()  # make a copy of the original dataset
        burst['time'] = burst['time'].dt.round('H')  # reset the time values to the nearest hour
        burst = burst.resample(time='1H', skipna=True).median(dim='time', keep_attrs=True)
        burst = burst.where(~np.isnan(burst.deployment), drop=True)

        # reset the attributes...which keep_attrs should do...
        burst.attrs = ds.attrs
        for v in burst.variables:
            burst[v].attrs = ds[v].attrs

        # save the newly average data
        ds = burst

    return ds


def median_absolute_difference(x):
    """
    Calculate the median absolute deviation

    :param x: array of values
    :return: median absolute deviation
    """
    # Calculate the median
    median = np.nanmedian(x)
    # Calculate the differences
    diffs = np.abs(x - median)
    # Calculate the median absolute difference
    mad = np.median(diffs)

    return mad


def quality_checks(ds, param, fail_min, fail_max, window="12H", center=True):
    """
    Checks the data for re-zeroing error which generates an anomalous low
    seawater pCO2 measurement. This functions calculates the median-absolute
    deviation on a 24-hour centered-rolling window. This is to identify the
    outliers

    :param ds: xarray dataset
    :param param: parameter to check
    :param fail_min: minimum value for the parameter
    :param fail_max: maximum value for the parameter
    :param window: window size for the rolling window
    :param center: center the rolling window
    :return: xarray dataset with the quality flag added
    """
    # First, nan out the out-of-range values
    mask = (ds[param] < fail_min) | (ds[param] > fail_max)
    ds[param][mask] = np.nan

    # Next, calculate the rolling mean & standard deviation, interpolating missing values
    df = ds[param].to_dataframe().interpolate()
    median = df.rolling(window=window, center=center).median()
    mad = df.rolling(window=window, center=center).apply(median_absolute_difference)

    # Create a flag array to store the results
    quality_flag = ds[param].astype(int) * 0 + 1

    # Identify where values are below the 3 standard deviations
    mask = ds[param] < median[param] - 3 * mad[param]
    quality_flag[mask] = 3

    # Add to the dataset
    # Add the flags to the dataset
    ds[param + "_quality_flag"] = quality_flag

    # Add attributes
    ds[param+"_quality_flag"].attrs = {
        "standard_name": param+"_qc_flag",
        "long_name": f"QC Flag for {param}",
        "comment": "Flag which attempts to identify low-pCO2 values due to the re-zeroing sampling issue " +
                   "of the Pro-Oceanus with occurs about once every 12H."
    }

    return ds


def main(argv=None):
    # set up the input arguments
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    method = args.method
    stream = args.stream
    deploy = args.deploy
    start = args.start
    stop = args.stop
    burst = args.burst

    # determine the start and stop times for the data request based on either the deployment number or user entered
    # beginning and ending dates.
    if not deploy or (start and stop):
        return SyntaxError('You must specify either a deployment number or beginning and end dates of interest.')
    else:
        if deploy:
            # Determine start and end dates based on the deployment number
            start, stop = get_deployment_dates(site, node, sensor, deploy)
            if not start or not stop:
                exit_text = ('Deployment dates are unavailable for %s-%s-%s, deployment %02d.' % (site, node, sensor,
                                                                                                  deploy))
                raise SystemExit(exit_text)

    # Request the data for download
    r = m2m_request(site, node, sensor, method, stream, start, stop)
    if not r:
        exit_text = ('Request failed for %s-%s-%s. Check request.' % (site, node, sensor))
        raise SystemExit(exit_text)

    # Valid request, start downloading the data
    if re.match(r'.*_air.*', stream):
        if deploy:
            pco2a = m2m_collect(r, ('.*deployment%04d.*PCO2A.*air.*\\.nc$' % deploy))
        else:
            pco2a = m2m_collect(r, '.*PCO2A.*air.*\\.nc$')
        nc_group = 'air'
    else:
        if deploy:
            pco2a = m2m_collect(r, ('.*deployment%04d.*PCO2A.*water.*\\.nc$' % deploy))
        else:
            pco2a = m2m_collect(r, '.*PCO2A.*water.*\\.nc$')
        nc_group = 'water'

    if not pco2a:
        exit_text = ('Data unavailable for %s-%s-%s. Check request.' % (site, node, sensor))
        raise SystemExit(exit_text)

    # clean-up and reorganize
    pco2a = pco2a_datalogger(pco2a, burst)
    vocab = get_vocabulary(site, node, sensor)[0]
    pco2a = update_dataset(pco2a, vocab['maxdepth'])

    # save the data to disk
    out_file = os.path.abspath(args.outfile)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    if os.path.isfile(out_file):
        pco2a.to_netcdf(out_file, mode='a', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS, group=nc_group)
    else:
        pco2a.to_netcdf(out_file, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS, group=nc_group)


if __name__ == '__main__':
    main()
