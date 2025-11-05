#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import re
import sys
import time
import xarray as xr

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from ooi_data_explorations.common import inputs, m2m_request, list_files, m2m_collect, \
    load_gc_thredds, get_vocabulary, update_dataset, N_CORES, ENCODINGS
from ooi_data_explorations.profilers import create_profile_id, bin_profiles, updown
from ooi_data_explorations.uncabled.process_optaa import ATTRS, FILL_INT, load_cal_coefficients, apply_dev, \
    apply_tscorr, apply_scatcorr, estimate_chl_poc, calculate_ratios

from pyseas.data.opt_functions import opt_internal_temp, opt_external_temp

# set floating-point error handling for numpy
_ = np.seterr(all='ignore', divide='warn')


def optaa_benthic(ds, cal_file):
    """
    Takes OPTAA data streamed to shore from the Cabled Array benthic platforms
    and cleans up the data set to make it more user-friendly. Primary task is
    renaming parameters and dropping some that are of limited use.
    Additionally, re-calculate the intermediate products (e.g. absorption and
    attenuation) and add them to the data set. Finally, add the estimated
    chlorophyll and POC concentrations to the data set.

    Will test the data set to determine if more than one deployment is present.
    If so, will raise an exception with an error message. AC-S processing
    requires that the data be processed one deployment at a time in order to
    properly assign calibration coefficients and pad wavelength arrays.

    :param ds: initial optaa data set downloaded from OOI via the M2M system
    :param cal_file: file name (can include path) to store the calibration
        coefficients
    :return ds: cleaned up and reprocessed data set
    """
    # check to see if there is more than one deployment in the data set
    if len(np.unique(ds['deployment'].values)) > 1:
        raise ValueError('More than one deployment in the data set.  Please structure processing request to process '
                         'one deployment at a time.')

    # drop some of the variables:
    #   internal_timestamp == time, redundant so can remove
    #   pressure_counts == none of the OOI OPTAAs have a pressure sensor
    #   serial_number == available in the global attributes
    #   meter_type == always the same, not needed
    #   packet_type == always the same, not needed
    #   record_length == always the same, not needed
    #   checksum == not needed, used in data parsing
    ds = ds.drop_vars(['internal_timestamp', 'pressure_counts', 'serial_number', 'meter_type', 'packet_type',
                  'record_length', 'checksum'])

    # check for data from a co-located CTD, if not present create the variables using NaN's as the fill value
    if 'sea_water_temperature' not in ds.variables:
        ds['sea_water_temperature'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_practical_salinity'] = ('time', ds['deployment'].data * np.nan)

    # pull out the number of wavelengths and serial number and then drop the variables (part of the metadata)
    num_wavelengths = ds.num_wavelengths.values[0].astype(int)
    serial_number = int(re.sub('[^0-9]', '', ds.attrs['SerialNumber']))
    ds = ds.drop_vars('num_wavelengths')

    # load the calibration coefficients
    uid = ds.attrs['AssetUniqueID']
    start_time = ds['time'][0].values.astype(float) / 10 ** 9
    cal = load_cal_coefficients(cal_file, uid, start_time)

    # check the calibration coefficients against the deployment data
    if cal.coeffs['serial_number'] != serial_number:
        raise Exception('Serial Number mismatch between ac-s data and the device file.')
    if cal.coeffs['num_wavelengths'] != num_wavelengths:
        raise Exception('Number of wavelengths mismatch between ac-s data and the device file.')

    # remove the units from the variable names
    rename = {
        'a_signal_dark_counts': 'a_signal_dark',
        'a_reference_dark_counts': 'a_reference_dark',
        'a_signal_counts': 'a_signal',
        'a_reference_counts': 'a_reference',
        'c_signal_dark_counts': 'c_signal_dark',
        'c_reference_dark_counts': 'c_reference_dark',
        'c_signal_counts': 'c_signal',
        'c_reference_counts': 'c_reference',
        'wavelength': 'wavelength_number'
    }
    ds = ds.rename(rename)

    # Delete the first 45 seconds of the data record per recommendation from the vendor. Note, originally the vendor
    # recommended deleting the first 45 seconds, then 60 seconds and then 120 seconds.  They never provided a data
    # based reason for the change in recommendation. Within OOI, instruments were programmed to run for 60 seconds,
    # then 120 seconds and then 240 seconds ... and it is all mixed up across the various data sets.  So, we are
    # going to use the 45-second recommendation and apply it to all data sets. If the vendor ever provides an analysis
    # justifying the change in recommendation, we can revisit this.
    ds.elapsed_run_time.values = ds.elapsed_run_time.where(ds.elapsed_run_time / 1000 > 45)
    ds = ds.dropna(dim='time', subset=['elapsed_run_time'])

    # convert internal and external temperature sensors from raw counts to degrees Celsius
    ds['internal_temp'] = opt_internal_temp(ds['internal_temp_raw'])
    ds['external_temp'] = opt_external_temp(ds['external_temp_raw'])

    # calculate the median of the remaining data per burst measurement (configured to run hourly for 3 minutes)
    # calculate the median of the remaining data per burst measurement
    print('Calculating burst averages ...')
    start_time = time.time()
    burst = ds.resample(time='3600s', base=1800, loffset='1800s', skipna=True).reduce(np.median, dim='time',
                                                                                      keep_attrs=True)
    burst = burst.where(~np.isnan(burst.deployment), drop=True)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print('... burst averaging complete.  Elapsed time: %f seconds' % elapsed_time)

    # re-process the raw data in order to create the intermediate variables, correcting for the holographic
    # grating, applying the temperature and salinity corrections and applying a baseline scatter correction
    # to the absorption data. All intermediate processing outputs are added to the data set.
    burst = apply_dev(burst, cal.coeffs)
    burst = apply_tscorr(burst, cal.coeffs, burst.sea_water_temperature, burst.sea_water_practical_salinity)
    burst = apply_scatcorr(burst, cal.coeffs)

    # add the jump offsets as NaN's if the grating index correction was not used
    if 'a_jump_offsets' not in ds.variables:
        ds['a_jump_offsets'] = ('time', ds['deployment'].data * np.nan)
        ds['c_jump_offsets'] = ('time', ds['deployment'].data * np.nan)

    # estimate chlorophyll and POC and calculate select absorption ratios
    burst = estimate_chl_poc(burst, cal.coeffs)
    burst = calculate_ratios(burst)

    # create a xarray dataset of the 2D variables, padding the number of wavelengths to a consistent
    # length of 100 using fill values.
    wavelength_number = np.arange(100).astype(int)  # used as a dimensional variable
    pad = 100 - num_wavelengths
    fill_nan = np.tile(np.ones(pad) * np.nan, (len(burst.time), 1))
    fill_int = np.tile(np.ones(pad) * FILL_INT, (len(burst.time), 1))

    wavelength_a = np.concatenate([burst.wavelength_a.values, fill_nan], axis=1)
    wavelength_c = np.concatenate([burst.wavelength_c.values, fill_nan], axis=1)

    ac = xr.Dataset({
        'wavelength_a': (['time', 'wavelength_number'], wavelength_a),
        'a_signal': (['time', 'wavelength_number'], np.concatenate([burst.a_signal, fill_int], axis=1).astype(int)),
        'a_reference': (['time', 'wavelength_number'], np.concatenate([burst.a_reference, fill_int],
                                                                      axis=1).astype(int)),
        'optical_absorption': (['time', 'wavelength_number'], np.concatenate([burst.optical_absorption, fill_nan],
                                                                             axis=1)),
        'apg': (['time', 'wavelength_number'], np.concatenate([burst.apg, fill_nan], axis=1)),
        'apg_ts': (['time', 'wavelength_number'], np.concatenate([burst.apg_ts, fill_nan], axis=1)),
        'apg_ts_s': (['time', 'wavelength_number'], np.concatenate([burst.apg_ts_s, fill_nan], axis=1)),
        'wavelength_c': (['time', 'wavelength_number'], wavelength_c),
        'c_signal': (['time', 'wavelength_number'], np.concatenate([burst.c_signal, fill_int], axis=1).astype(int)),
        'c_reference': (['time', 'wavelength_number'], np.concatenate([burst.c_reference, fill_int],
                                                                      axis=1).astype(int)),
        'beam_attenuation': (['time', 'wavelength_number'], np.concatenate([burst.beam_attenuation, fill_nan], axis=1)),
        'cpg': (['time', 'wavelength_number'], np.concatenate([burst.cpg, fill_nan], axis=1)),
        'cpg_ts': (['time', 'wavelength_number'], np.concatenate([burst.cpg_ts, fill_nan], axis=1)),
    }, coords={'time': (['time'], burst.time.values), 'wavelength_number': wavelength_number})

    # drop the original 2D variables from the burst data set
    drop = burst.drop_vars(['wavelength_number', 'wavelength_a', 'a_signal', 'a_reference',
                       'optical_absorption', 'apg', 'apg_ts', 'apg_ts_s',
                       'wavelength_c', 'c_signal', 'c_reference',
                       'beam_attenuation', 'cpg', 'cpg_ts'])

    # reset the data type for the 'a' and 'c' signal and reference dark values, and the other raw parameters
    int_arrays = ['a_signal_dark', 'a_reference_dark', 'c_signal_dark', 'c_reference_dark',
                  'internal_temp_raw', 'external_temp_raw', 'deployment']
    for k in drop.variables:
        if k in int_arrays:
            drop[k] = drop[k].astype(int)

    # recombine the two datasets
    optaa = xr.merge([drop, ac])

    # reset the attributes, which the merging drops
    optaa.attrs = burst.attrs
    for v in optaa.variables:
        optaa[v].attrs = burst[v].attrs

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in optaa.variables:
                optaa[key].attrs[atk] = atv

    # add the original variable name as an attribute, if renamed
    for key, value in rename.items():
        optaa[value].attrs['ooinet_variable_name'] = key

    # add the actual number of wavelengths to the dataset as an attribute
    optaa['wavelength_number'].attrs['actual_wavelengths'] = num_wavelengths

    # if the filter index was used to adjust the spectral jumps, add that attribute to the data set
    if cal.coeffs['grate_index']:
        optaa['a_jump_offsets'].attrs['grate_index'] = cal.coeffs['grate_index']
        optaa['c_jump_offsets'].attrs['grate_index'] = cal.coeffs['grate_index']

    return optaa


def optaa_profiler(ds, cal_file):
    """
    Takes OPTAA data recorded by the Cabled Shallow Profiler system and cleans
    up the data set to make it more user-friendly.  Primary task is renaming
    parameters and dropping some that are of limited use.  Additionally,
    re-calculate the intermediate products (e.g. absorption and attenuation)
    and add them to the data set.  Finally, add the estimated chlorophyll and
    POC concentrations to the data set.

    Will test the data set to determine if more than one deployment is present.
    If so, will raise an exception with an error message. AC-S processing
    requires that the data be processed one deployment at a time in order to
    properly assign calibration coefficients and pad wavelength arrays.

    :param ds: initial optaa data set downloaded from OOI via the M2M system
    :param cal_file: file name (can include path) to store the calibration
        coefficients
    :return ds: cleaned up and reprocessed data set
    """
    # check to see if there is more than one deployment in the data set
    if len(np.unique(ds['deployment'].values)) > 1:
        raise ValueError('More than one deployment in the data set.  Please structure processing request to process '
                         'one deployment at a time.')

    # drop some of the variables:
    #   internal_timestamp == time, redundant so can remove
    #   pressure_counts == none of the OOI OPTAAs have a pressure sensor
    #   serial_number == available in the global attributes
    #   meter_type == always the same, not needed
    #   packet_type == always the same, not needed
    #   record_length == always the same, not needed
    #   checksum == not needed, used in data parsing
    ds = ds.drop_vars(['internal_timestamp', 'pressure_counts', 'serial_number', 'meter_type', 'packet_type',
                  'record_length', 'checksum'])

    # check for data from a co-located CTD, if not present create the variables using NaN's as the fill value
    if 'sea_water_temperature' not in ds.variables:
        ds['sea_water_temperature'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_practical_salinity'] = ('time', ds['deployment'].data * np.nan)

    # pull out the number of wavelengths and serial number and then drop the variable (part of the metadata)
    num_wavelengths = ds.num_wavelengths.values[0].astype(int)
    serial_number = int(re.sub('[^0-9]', '', ds.attrs['SerialNumber']))
    ds = ds.drop_vars('num_wavelengths')

    # load the calibration coefficients
    uid = ds.attrs['AssetUniqueID']
    start_time = ds['time'][0].values.astype(float) / 10 ** 9
    cal = load_cal_coefficients(cal_file, uid, start_time)

    # check the calibration coefficients against the deployment data
    if cal.coeffs['serial_number'] != serial_number:
        raise Exception('Serial Number mismatch between ac-s data and the device file.')
    if cal.coeffs['num_wavelengths'] != num_wavelengths:
        raise Exception('Number of wavelengths mismatch between ac-s data and the device file.')

    # remove the units from the variable name
    rename = {
        'a_signal_dark_counts': 'a_signal_dark',
        'a_reference_dark_counts': 'a_reference_dark',
        'a_signal_counts': 'a_signal',
        'a_reference_counts': 'a_reference',
        'c_signal_dark_counts': 'c_signal_dark',
        'c_reference_dark_counts': 'c_reference_dark',
        'c_signal_counts': 'c_signal',
        'c_reference_counts': 'c_reference',
        'int_ctd_pressure': 'sea_water_pressure',
        'wavelength': 'wavelength_number'
    }
    ds = ds.rename(rename)

    # Delete the first 45 seconds of the data record per recommendation from the vendor. Note, originally the vendor
    # recommended deleting the first 45 seconds, then 60 seconds and then 120 seconds.  They never provided a data
    # based reason for the change in recommendation. Within OOI, instruments were programmed to run for 60 seconds,
    # then 120 seconds and then 240 seconds ... and it is all mixed up across the various data sets.  So, we are
    # going to use the 45-second recommendation and apply it to all data sets. If the vendor ever provides an analysis
    # justifying the change in recommendation, we can revisit this.
    ds.elapsed_run_time.values = ds.elapsed_run_time.where(ds.elapsed_run_time / 1000 > 45)
    ds = ds.dropna(dim='time', subset=['elapsed_run_time'])

    # convert internal and external temperature sensors from raw counts to degrees Celsius
    ds['internal_temp'] = opt_internal_temp(ds['internal_temp_raw'])
    ds['external_temp'] = opt_external_temp(ds['external_temp_raw'])

    # create a profile variable to uniquely identify profiles within the dataset...if the profiler moved
    print('Determining profiler movement ...')
    pks, dzdt = updown(ds['depth'].values, 10)
    if len(pks) == 2 and (pks[0] == 0 and pks[1] == len(dzdt) - 1):
        # the profiler never moved, so treat the whole data set as a time series
        print('Profiler was parked for the entire deployment, treating data as a time series using burst averaging.')
        ds['profile'] = ('time', np.zeros(len(ds['time'])).astype(int) - 1)

        # calculate the median of the remaining data per burst measurement (configured to run hourly for 3 minutes)
        print('Calculating burst averages ...')
        start_time = time.time()
        binned = ds.resample(time='3600s', base=1800, loffset='1800s', skipna=True).reduce(np.median, dim='time',
                                                                                           keep_attrs=True)
        binned = binned.where(~np.isnan(binned.deployment), drop=True)
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        print('... burst averaging complete.  Elapsed time: %f seconds' % elapsed_time)
    else:
        # the profiler moved, so treat the data as a series of profiles
        print('Profiler moved during the deployment, treating data as a series of profiles.')
        print('Sub-selecting upcast data only from the data set ...')
        dzdt = xr.DataArray(dzdt, dims='time', coords={'time': ds['time']})
        ds = ds.where(dzdt < 0, drop=True)
        print('Creating and adding a profile variable to the data set ...')
        ds = create_profile_id(ds)

        # group the data by profile number and bin the data into 25 cm depth bins (nominal ascent rate of the shallow
        # profiler is 5 cm/s, binning at 25 cm will help to reduce the noise in the data and speed up subsequent
        # processing).
        profiles = ds.groupby('profile')
        profiles = [profile[1] for profile in profiles]
        partial_binning = partial(bin_profiles, site_depth=200, bin_size=0.25)
        with ProcessPoolExecutor(max_workers=N_CORES) as executor:
            binned = list(tqdm(executor.map(partial_binning, profiles), total=len(profiles),
                               desc='Smoothing and binning each profile into 25 cm depth bins ...', file=sys.stdout))

        # reset the dataset now using binned profiles
        binned = [i for i in binned if i is not None]
        binned = xr.concat(binned, 'time')
        binned = binned.sortby(['profile', 'time'])

        # confirm dimension order is correct for the wavelength arrays (sometimes the order gets flipped
        # during the binning process)
        binned['wavelength_a'] = binned.wavelength_a.transpose(*['time', 'wavelength_number'])
        binned['wavelength_c'] = binned.wavelength_c.transpose(*['time', 'wavelength_number'])

    # reclaim some memory
    del ds, pks, dzdt, profiles, partial_binning, executor

    # re-process the raw data in order to create the intermediate variables, correcting for the holographic
    # grating, applying the temperature and salinity corrections and applying a baseline scatter correction
    # to the absorption data. All intermediate processing outputs are added to the data set.
    print('Re-processing the raw data, creating intermediate data products ...')
    binned = apply_dev(binned, cal.coeffs)
    binned = apply_tscorr(binned, cal.coeffs, binned.sea_water_temperature, binned.sea_water_practical_salinity)
    binned = apply_scatcorr(binned, cal.coeffs)

    # estimate chlorophyll and POC and calculate select absorption ratios
    binned = estimate_chl_poc(binned, cal.coeffs)
    binned = calculate_ratios(binned)

    # create a xarray dataset of the 2D variables, padding the number of wavelengths to a consistent
    # length of 100 using fill values.
    wavelength_number = np.arange(100).astype(int)  # used as a dimensional variable
    pad = 100 - num_wavelengths
    fill_nan = np.tile(np.ones(pad) * np.nan, (len(binned.time), 1))
    fill_int = np.tile(np.ones(pad) * FILL_INT, (len(binned.time), 1))

    wavelength_a = np.concatenate([binned.wavelength_a.values, fill_nan], axis=1)
    wavelength_c = np.concatenate([binned.wavelength_c.values, fill_nan], axis=1)

    ac = xr.Dataset({
        'wavelength_a': (['time', 'wavelength_number'], wavelength_a),
        'a_signal': (['time', 'wavelength_number'], np.concatenate([binned.a_signal, fill_int], axis=1).astype(int)),
        'a_reference': (['time', 'wavelength_number'], np.concatenate([binned.a_reference, fill_int],
                                                                      axis=1).astype(int)),
        'optical_absorption': (['time', 'wavelength_number'], np.concatenate([binned.optical_absorption, fill_nan],
                                                                             axis=1)),
        'apg': (['time', 'wavelength_number'], np.concatenate([binned.apg, fill_nan], axis=1)),
        'apg_ts': (['time', 'wavelength_number'], np.concatenate([binned.apg_ts, fill_nan], axis=1)),
        'apg_ts_s': (['time', 'wavelength_number'], np.concatenate([binned.apg_ts_s, fill_nan], axis=1)),
        'wavelength_c': (['time', 'wavelength_number'], wavelength_c),
        'c_signal': (['time', 'wavelength_number'], np.concatenate([binned.c_signal, fill_int], axis=1).astype(int)),
        'c_reference': (['time', 'wavelength_number'], np.concatenate([binned.c_reference, fill_int],
                                                                      axis=1).astype(int)),
        'beam_attenuation': (['time', 'wavelength_number'], np.concatenate([binned.beam_attenuation, fill_nan],
                                                                           axis=1)),
        'cpg': (['time', 'wavelength_number'], np.concatenate([binned.cpg, fill_nan], axis=1)),
        'cpg_ts': (['time', 'wavelength_number'], np.concatenate([binned.cpg_ts, fill_nan], axis=1)),
    }, coords={'time': (['time'], binned.time.values), 'wavelength_number': wavelength_number})

    # drop the original 2D variables from the binned data set
    drop = binned.drop_vars(['wavelength_number', 'wavelength_a', 'a_signal', 'a_reference',
                        'optical_absorption', 'apg', 'apg_ts', 'apg_ts_s',
                        'wavelength_c', 'c_signal', 'c_reference',
                        'beam_attenuation', 'cpg', 'cpg_ts'])

    # reset the data type for the 'a' and 'c' signal and reference dark values, and the other raw parameters
    int_arrays = ['a_signal_dark', 'a_reference_dark', 'c_signal_dark', 'c_reference_dark',
                  'internal_temp_raw', 'external_temp_raw', 'deployment', 'profile']
    for k in drop.variables:
        if k in int_arrays:
            drop[k] = drop[k].astype(int)

    # recombine the two datasets
    optaa = xr.merge([drop, ac])

    # reset the attributes, which the merging drops
    optaa.attrs = binned.attrs
    for v in optaa.variables:
        optaa[v].attrs = binned[v].attrs

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in optaa.variables:
                optaa[key].attrs[atk] = atv

    # add the original variable name as an attribute, if renamed
    for key, value in rename.items():
        optaa[value].attrs['ooinet_variable_name'] = key

    # add the actual number of wavelengths to the dataset as an attribute
    optaa['wavelength_number'].attrs['actual_wavelengths'] = num_wavelengths

    # if the filter index was used to adjust the spectral jumps, add that attribute to the data set
    if cal.coeffs['grate_index']:
        optaa['a_jump_offsets'].attrs['grate_index'] = cal.coeffs['grate_index']
        optaa['c_jump_offsets'].attrs['grate_index'] = cal.coeffs['grate_index']

    return optaa


def main(argv=None):
    """
    Command line interface for processing OOI OPTAA NetCDF file(s) from the
    Cabled Array benthic or shallow profiler platforms. Creates a cleaned and
    processed xarray dataset of the OPTAA data saved to a NetCDF file.
    """
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    method = args.method
    stream = args.stream
    deploy = args.deploy
    start = args.start
    stop = args.stop

    # check if we are specifying a deployment or a specific date and time range
    if not deploy or (start and stop):
        return SyntaxError('You must specify either a deployment number or beginning and end dates of interest.')

    # if we are specifying a deployment number, then get the data from the Gold Copy THREDDS server
    if deploy:
        optaa = load_gc_thredds(site, node, sensor, method, stream, ('.*deployment%04d.*OPTAA.*\\.nc$' % deploy))
        cal_file = ('{}-{}-{}.deploy{:02d}.cal_coeffs.json'.format(site, node, sensor, deploy))

        # check to see if we downloaded any data
        if not optaa:
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

        # OPTAA data is different from other instruments. it needs to be processed on a per-deployment basis in order
        # to get the correct number of wavelengths before it can be merged into a single dataset. create a list of
        # all the files that were returned by the M2M request, and determine the deployments that are included in the
        # request
        files = list_files(r['allURLs'][0], '.+OPTAA.+\\.nc$')
        if not files:
            exit_text = ('Data unavailable for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                    stream, start, stop))
            raise SystemExit(exit_text)

        deployments = np.unique([int(sub.split('/')[3][10:14]) for sub in files])

        # loop through the deployments and download the data for each one
        optaa = []
        cal_file = []
        for deploy in deployments:
            # Valid M2M request, download the data on a per-deployment basis
            data = m2m_collect(r, ('.*deployment%04d.*OPTAA.*\\.nc$' % deploy))
            if data:
                optaa.append(data)
                cal_file.append('{}-{}-{}.deploy{:02d}.cal_coeffs.json'.format(site, node, sensor, deploy))

        # check to see if we downloaded any data (remove empty/none entries from the list)
        if not optaa:
            exit_text = ('Data unavailable for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                    stream, start, stop))
            raise SystemExit(exit_text)

    # set up the calibration file path and name(s)
    out_file = os.path.abspath(args.outfile)
    cal_path = os.path.dirname(out_file)
    if not os.path.exists(cal_path):
        os.makedirs(cal_path)

    # clean-up and reorganize the data
    multi = isinstance(optaa, list)
    if node in ['SF01A', 'SF01B', 'SF03A']:
        # this OPTAA is on a shallow profiler
        if multi:
            for i, ds in enumerate(optaa):
                cfile = os.path.join(cal_path, cal_file[i])
                optaa[i] = optaa_profiler(ds, cfile)
            optaa = xr.concat(optaa, dim='time')
        else:
            cal_file = os.path.join(cal_path, cal_file)
            optaa = optaa_profiler(optaa, cal_file)
    else:
        # this OPTAA is on one of the two benthic platforms
        if multi:
            for i, ds in enumerate(optaa):
                cfile = os.path.join(cal_path, cal_file[i])
                optaa[i] = optaa_benthic(ds, cfile)
            optaa = xr.concat(optaa, dim='time')
        else:
            optaa = optaa_benthic(optaa, cal_file)

    # get the vocabulary information for the site, node, and sensor and update the dataset attributes
    vocab = get_vocabulary(site, node, sensor)[0]
    optaa = optaa.sortby(['deployment', 'time'])
    optaa = update_dataset(optaa, vocab['maxdepth'])

    # save the data to disk
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    optaa.to_netcdf(out_file, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)


if __name__ == '__main__':
    main()
