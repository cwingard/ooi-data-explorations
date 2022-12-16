#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os

from ooi_data_explorations.common import inputs, m2m_collect, m2m_request, load_gc_thredds, \
    get_vocabulary, update_dataset, ENCODINGS
from ooi_data_explorations.qartod.qc_processing import parse_qc

from gsw.conversions import SP_from_C
from ioos_qc import qartod


def quality_checks(ds):
    """
    Assignment of QARTOD style quality flags to the 1-minute resolution METBK
    (bulk meteorological) data on a per-parameter basis. Two tests are
    performed: the IOOS QARTOD flat line test for all bulk parameters, and a
    fill value (missing) test. Results are represented using a subset of the
    QARTOD flags to indicate the quality. QARTOD flags used are:

        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail
        9 = Missing

    The final flag value represents the worst case assessment of the data
    quality.

    :param ds: xarray dataset with the 1-minute resolution METBK data
    :return ds: dataset with QARTOD style quality flags added to the record
        per variable tested.
    """
    parameters = ['barometric_pressure', 'relative_humidity', 'air_temperature', 'longwave_irradiance',
                  'precipitation', 'shortwave_irradiance', 'sea_surface_temperature', 'sea_surface_conductivity',
                  'eastward_wind_velocity', 'northward_wind_velocity']
    for p in parameters:
        # The primary failure mode of the METBK is to repeat the last value it received from a sensor.
        # Use the IOOS QARTOD flat line test to identify these cases (consider it suspect if it repeats
        # for 5+ minutes and failed if it repeats for 15+ minutes).
        flags = qartod.flat_line_test(ds[p].values, ds['time'].values, 300, 900, 0.01)

        # The secondary failure mode occurs when the METBK logger sets values to a NaN if no sensor data is available.
        # In the case of the sea surface conductivity and temperature data, different values are used to represent
        # missing data. Specifically, the values are set to a 0.0 and -5.0, respectively. In either case, (NaNs or
        # 0.0 and -5.0) set the QC flag to 9 to indicate "Missing" data, and then convert the 0.0 and -5.0 values to
        # a NaN to avoid propagating false numbers into subsequent calculations (e.g. salinity or heat flux).
        if p == 'sea_surface_temperature':
            m = ds[p] < -4.0  # use a floating point value just above -5
            flags[m] = 9
            ds[p][m] = np.nan
        elif p == 'sea_surface_conductivity':
            m = ds[p] < 0.5  # use a floating point value just above 0
            flags[m] = 9
            ds[p][m] = np.nan
        else:
            m = np.isnan(ds[p])
            flags[m] = 9

        # add the qc_flags to the dataset, rolling up the results into a single value
        qc_summary = p + '_qc_summary_flag'
        if qc_summary in ds.variables:
            # add the new test results to the existing QC summary results
            qc = ds[qc_summary]
            flags = np.array([flags, qc.values])
            ds[qc_summary] = ('time', flags.max(axis=0))
        else:
            # create a new QC summary variable
            ds[qc_summary] = ('time', flags)

            # set up the attributes for the new variable
            ds[qc_summary].attrs = dict({
                'long_name': '%s QC Summary Flag' % ds[p].attrs['long_name'],
                'comment': ('Converts the QC Results values from a bitmap to a QARTOD style summary flag, where ',
                            'the values are 1 == pass, 2 == not evaluated, 3 == suspect or of high interest, ',
                            '4 == fail, and 9 == missing. The QC tests, as applied by OOI, only yield pass or ',
                            'fail values. By resetting, the QC flags become more user friendly and more nuanced.'),
                'flag_values': np.array([1, 2, 3, 4, 9]),
                'flag_meanings': 'pass not_evaluated suspect_or_of_high_interest fail missing'
            })

            # add the standard name if the variable has one
            if 'standard_name' in ds[p].attrs:
                ds[qc_summary].attrs['standard_name'] = '%s qc_summary_flag' % ds[p].attrs['standard_name']

        return ds


def metbk_hourly(ds):
    """
    Takes METBK hourly averaged bulk flux estimates from the CGSN/EA moorings
    and cleans up the data set to make it more user-friendly. Primary task is
    renaming parameters and dropping some that are of limited use.
    Additionally, re-organize some variables to permit better assessments of
    the data.

    :param ds: initial metbk hourly averaged data set downloaded from OOI via
        the M2M system
    :return ds: cleaned up data set
    """
    # drop some variables:
    #   met_timeflx == time, redundant,
    #   ### Data products from upstream processing used to calculate hourly flux measurements. Remove from here to
    #   ### keep this data set clean. Will obtain the 1-minute source data from a separate stream.
    #   eastward_velocity
    #   northward_velocity
    #   longwave_irradiance
    #   air_temperature
    #   barometric_pressure
    #   precipitation
    #   sea_surface_temperature
    #   relative_humidity
    #   shortwave_irradiance
    ds = ds.drop(['met_timeflx', 'eastward_velocity', 'northward_velocity', 'longwave_irradiance', 'air_temperature',
                  'barometric_pressure', 'precipitation', 'sea_surface_temperature', 'relative_humidity',
                  'shortwave_irradiance'])

    # reset incorrectly formatted temperature units
    temp_vars = ['met_tempa2m', 'met_tempskn']
    for var in temp_vars:
        ds[var].attrs['units'] = 'degree_Celsius'

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    return ds


def metbk_datalogger(ds, burst=False):
    """
    Takes METBK data recorded by the data loggers used in the CGSN/EA moorings
    and cleans up the data set to make it more user-friendly.  Primary task is
    renaming parameters and dropping some that are of limited use.
    Additionally, re-organize some variables to permit better assessments of
    the data.

    :param ds: initial metbk data set downloaded from OOI via the M2M system
    :param burst: resample the 1-minute data to a 15-minute time interval
    :return ds: cleaned up data set
    """
    # drop some variables:
    #   date_time_string == internal_timestamp, redundant so can remove
    #   dcl_controller_timestamp == time, redundant so can remove
    #   internal_timestamp == doesn't exist, always empty so can remove
    #   ### Data products from downstream processing used to calculate hourly flux measurements. Remove from here to
    #   ### keep this data set clean. Will obtain hourly flux data from a different stream.
    #   met_barpres
    #   met_windavg_mag_corr_east
    #   met_windavg_mag_corr_north
    #   met_netsirr
    #   met_salsurf
    #   met_spechum
    #   ct_depth
    #   met_current_direction
    #   met_current_speed
    #   met_relwind_direction
    #   met_relwind_speed
    #   met_heatflx_minute
    #   met_latnflx_minute
    #   met_netlirr_minute
    #   met_sensflx_minute
    ds = ds.drop(['dcl_controller_timestamp', 'internal_timestamp', 'met_barpres', 'met_windavg_mag_corr_east',
                  'met_windavg_mag_corr_north', 'met_netsirr', 'met_spechum', 'ct_depth', 'met_current_direction',
                  'met_current_speed', 'met_relwind_direction', 'met_relwind_speed', 'met_heatflx_minute',
                  'met_latnflx_minute', 'met_netlirr_minute', 'met_sensflx_minute', 'met_barpres_qc_executed',
                  'met_barpres_qc_results', 'met_current_direction_qc_executed', 'met_current_direction_qc_results',
                  'met_current_speed_qc_executed', 'met_current_speed_qc_results', 'met_relwind_direction_qc_executed',
                  'met_relwind_direction_qc_results', 'met_relwind_speed_qc_executed', 'met_relwind_speed_qc_results',
                  'met_netsirr_qc_executed', 'met_netsirr_qc_results', 'met_spechum_qc_executed',
                  'met_spechum_qc_results'])

    # reset incorrectly formatted temperature and relative humidity units
    ds['relative_humidity'].attrs['units'] = 'percent'
    temp_vars = ['air_temperature', 'sea_surface_temperature']
    for var in temp_vars:
        ds[var].attrs['units'] = 'degree_Celsius'

    # rename the met_salsurf parameter
    rename = {
        'met_salsurf_qartod_executed': 'sea_surface_salinity_qartod_executed',
        'met_salsurf_qartod_results': 'sea_surface_salinity_qartod_results'
    }
    for key, value in rename.items():
        if key in ds.variables:
            ds = ds.rename({key: value})
            ds[value].attrs['ooinet_variable_name'] = key

    # correct the flag_values attribute for the *_qartod_results variables
    for qresult in ['sea_surface_conductivity_qartod_results', 'sea_surface_temperature_qartod_results',
                    'sea_surface_salinity_qartod_results']:
        ds[qresult].attrs['flag_values'] = np.array([1, 2, 3, 4, 9])

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    # run quality checks, adding the results to the QC summary flag
    ds = quality_checks(ds)

    # re-calculate the near surface salinity using the now NaN'ed out fill-values
    ds['sea_surface_salinity'] = SP_from_C(ds['sea_surface_conductivity'] * 10, ds['sea_surface_temperature'], 1.0)
    ds['sea_surface_salinity'].attrs = {
        '_FillValue': np.nan
    }

    if burst:   # re-sample the data to a 15-minute interval using a median average
        burst = ds
        burst = burst.resample(time='900s', base=3150, loffset='450s', skipna=True).median(dim='time', keep_attrs=True)
        burst = burst.where(~np.isnan(burst.deployment), drop=True)

        # save the newly average data
        ds = burst

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

    # check if we are specifying a deployment or a specific date and time range
    if not deploy or (start and stop):
        return SyntaxError('You must specify either a deployment number or beginning and end dates of interest.')

    # if we are specifying a deployment number, then get the data from the Gold Copy THREDDS server
    if deploy:
        # download the data for the deployment
        if stream in ['metbk_a_dcl_instrument', 'metbk_a_dcl_instrument_recovered']:
            tag = ('.*deployment%04d.*METBK.*\\.nc$' % deploy)
        else:
            tag = ('.*deployment%04d.*METBK.*hourly.*\\.nc$' % deploy)

        metbk = load_gc_thredds(site, node, sensor, method, stream, tag)

        # check to see if we downloaded any data
        if not metbk:
            exit_text = ('Data unavailable for %s-%s-%s, %s, %s, deployment %d.' % (site, node, sensor, method,
                                                                                    stream, deploy))
            raise SystemExit(exit_text)
    else:
        # otherwise, request the data for download from OOINet via the M2M API using the specified dates
        r = m2m_request(site, node, sensor, method, stream, start, stop)
        if not r:
            exit_text = ('Request failed for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                  stream, start, stop))
            raise SystemExit(exit_text)

        # Valid M2M request, start downloading the data
        if stream in ['metbk_a_dcl_instrument', 'metbk_a_dcl_instrument_recovered']:
            tag = '.*METBK.*\\.nc$'
        else:
            tag = '.*METBK.*hourly.*\\.nc$'

        metbk = m2m_collect(r, tag)

        # check to see if we downloaded any data
        if not metbk:
            exit_text = ('Data unavailable for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                    stream, start, stop))
            raise SystemExit(exit_text)

    # clean-up and reorganize
    if stream in ['metbk_a_dcl_instrument', 'metbk_a_dcl_instrument_recovered']:
        metbk = metbk_datalogger(metbk, burst)
    else:
        metbk = metbk_hourly(metbk)

    # vocab = get_vocabulary(site, node, sensor)[0]
    metbk = update_dataset(metbk, 1.0)

    # save the data to disk
    out_file = os.path.abspath(args.outfile)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    metbk.to_netcdf(out_file, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)


if __name__ == '__main__':
    main()
