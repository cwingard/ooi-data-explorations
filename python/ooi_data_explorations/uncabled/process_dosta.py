#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os

from ooi_data_explorations.common import inputs, m2m_collect, m2m_request, load_gc_thredds, get_deployment_dates, \
    get_vocabulary, update_dataset, dt64_epoch, ENCODINGS
from ooi_data_explorations.qartod.qc_processing import parse_qc

ATTRS = {
    'raw_oxygen_concentration': {
        'long_name': 'Raw Dissolved Oxygen Concentration',
        'units': 'counts',
        'comment': ('The dissolved oxygen concentration measurement reported in counts. Converted to a dissolved '
                    'oxygen concentration measurement reported in umols/L by dividing by 10000 and subtracting 10.'),
        'data_product_identifier': 'DOCONCS-CNT_L0',
    },
    'oxygen_concentration': {
        'long_name': 'Dissolved Oxygen Concentration',
        'standard_name': 'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water',
        'units': 'umol L-1',
        'comment': ('Mole concentration of dissolved oxygen per unit volume, also known as Molarity, as measured by '
                    'an optode oxygen sensor. Computed on-board the sensor using internal calibration coefficients.'),
        'data_product_identifier': 'DOCONCS_L1',
    },
    'oxygen_saturation': {
        'long_name': 'Dissolved Oxygen Saturation',
        'units': 'percent',
        'comment': ('Oxygen saturation is the percentage of dissolved oxygen relative to the absolute solubility of '
                    'oxygen at a particular water temperature. Computed on-board the sensor using internal calibration '
                    'coefficients.'),
    },
    'optode_temperature': {
        'long_name': 'Optode Thermistor Temperature',
        'standard_name': 'temperature_of_sensor_for_oxygen_in_sea_water',
        'units': 'degrees_Celsius',
        'comment': ('Optode internal thermistor temperature used in calculation of the absolute oxygen ' 
                    'concentration. This is not the in-situ sea water temperature, though it will be very close.'),
        'ancillary_variables': 'raw_temperature',
    },
    'calibrated_phase': {
        'long_name': 'Calibrated Phase Difference',
        'units': 'degrees',
        'comment': ('The optode measures oxygen by exciting a special platinum porphyrin complex embedded in a '
                    'gas permeable foil with modulated blue light. The optode measures the phase shift of the '
                    'returned red light. By linearizing and temperature compensating, with an incorporated '
                    'temperature sensor, the absolute O2 concentration can be determined.'),
        'data_product_identifier': 'DOCONCS-VLT_L0',
    },
    'temp_compensated_phase': {
        'long_name': 'Temperature Compensated Calibrated Phase',
        'comment': 'Temperature compensated (using the temperature data from an onboard thermistor) calibrated phase '
                   'difference.',
        'units': 'degrees',
        'ancillary_variables': 'optode_temperature, calibrated_phase'
    },
    'raw_temperature': {
        'long_name': 'Raw Optode Thermistor Temperature',
        'units': 'mV',
        'comment': ('The optode includes an integrated internal thermistor to measure the temperature at '
                    'the sensing foil.'),
    },
    # co-located CTD data
    'sea_water_temperature': {
        'long_name': 'Sea Water Temperature',
        'standard_name': 'sea_water_temperature',
        'units': 'degrees_Celsius',
        'comment': ('Sea water temperature is the in situ temperature of the sea water. Measurements are from a '
                    'co-located CTD'),
        'data_product_identifier': 'TEMPWAT_L1'
    },
    'sea_water_pressure': {
        'long_name': 'Sea Water Pressure',
        'standard_name': 'sea_water_pressure_due_to_sea_water',
        'units': 'dbar',
        'comment': ('Sea Water Pressure refers to the pressure exerted on a sensor in situ by the weight of the ' 
                    'column of seawater above it. It is calculated by subtracting one standard atmosphere from the ' 
                    'absolute pressure at the sensor to remove the weight of the atmosphere on top of the water ' 
                    'column. The pressure at a sensor in situ provides a metric of the depth of that sensor. '
                    'Measurements are from a co-located CTD.'),
        'data_product_identifier': 'PRESWAT_L1'
    },
    'sea_water_practical_salinity': {
        'long_name': 'Sea Water Practical Salinity',
        'standard_name': 'sea_water_practical_salinity',
        'units': '1',
        'comment': ('Salinity is generally defined as the concentration of dissolved salt in a parcel of sea water. ' 
                    'Practical Salinity is a more specific unitless quantity calculated from the conductivity of ' 
                    'sea water and adjusted for temperature and pressure. It is approximately equivalent to Absolute ' 
                    'Salinity (the mass fraction of dissolved salt in sea water), but they are not interchangeable. '
                    'Measurements are from a co-located CTD.'),
        'data_product_identifier': 'PRACSAL_L2'
    },
    # --> derived values
    'svu_oxygen_concentration': {
        'long_name': 'Dissolved Oxygen Concentration',
        'standard_name': 'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water',
        'units': 'umol L-1',
        'comment': ('Mole concentration of dissolved oxygen per unit volume, also known as Molarity, as measured by '
                    'an optode oxygen sensor. Compares to the oxygen_concentration computed on-board the sensor, '
                    'but is recomputed using factory calibration coefficients, the calibrated phase values and '
                    'the optode thermistor temperature via the Stern-Volmer-Uchida equation.'),
        'data_product_identifier': 'DOCONCS_L1'
    },
    'oxygen_concentration_corrected': {
        'long_name': 'Corrected Dissolved Oxygen Concentration',
        'standard_name': 'moles_of_oxygen_per_unit_mass_in_sea_water',
        'units': 'umol kg-1',
        'comments': ('The dissolved oxygen concentration from the Stable Response Dissolved Oxygen Instrument is a '
                     'measure of the concentration of gaseous oxygen mixed in seawater. This data product corrects '
                     'the dissolved oxygen concentration for the effects of salinity, temperature, and pressure with '
                     'data from a co-located CTD.'),
        'data_product_identifier': 'DOXYGEN_L2'
    }
}


def dosta_datalogger(ds, burst=False):
    """
    Takes dosta data recorded by the data loggers used in the CGSN/EA moorings
    and cleans up the data set to make it more user-friendly. Primary task is
    renaming the alphabet soup parameter names and dropping some parameters
    that are of no use/value.

    :param ds: initial dosta data set downloaded from OOI via the M2M system
    :param burst: resample the data to the defined time interval
    :return ds: cleaned up data set
    """
    # drop some of the variables:
    #   dcl_controller_timestamp == time, redundant so can remove
    #   internal_timestamp, there is no internal instrument clock
    #   product_number, these are all 4831s and captured in global attributes
    #   estimated_oxygen_saturation_qc_executed, preliminary data product no QC tests should be applied
    #   estimated_oxygen_saturation_qc_results, preliminary data product no QC tests should be applied
    #   CGSN Update: Redo to make this a list comprehension so limited parameter datasets can also be processed
    drop_list = ['dcl_controller_timestamp', 'product_number', 'internal_timestamp',
                 'estimated_oxygen_saturation_qc_executed', 'estimated_oxygen_saturation_qc_results']
    for var in ds.variables:
        if var in drop_list:
            ds = ds.drop(var)

    # rename variables for better clarity
    rename = {
        'estimated_oxygen_concentration': 'oxygen_concentration',
        'estimated_oxygen_concentration_qc_executed': 'oxygen_concentration_qc_executed',
        'estimated_oxygen_concentration_qc_results': 'oxygen_concentration_qc_results',
        'estimated_oxygen_concentration_qartod_executed': 'oxygen_concentration_qartod_executed',
        'estimated_oxygen_concentration_qartod_results': 'oxygen_concentration_qartod_results',
        'estimated_oxygen_saturation': 'oxygen_saturation',
        'dosta_abcdjm_cspp_tc_oxygen': 'svu_oxygen_concentration',
        'dosta_abcdjm_cspp_tc_oxygen_qc_executed': 'svu_oxygen_concentration_qc_executed',
        'dosta_abcdjm_cspp_tc_oxygen_qc_results': 'svu_oxygen_concentration_qc_results',
        'dosta_abcdjm_cspp_tc_oxygen_qartod_executed': 'svu_oxygen_concentration_qartod_executed',
        'dosta_abcdjm_cspp_tc_oxygen_qartod_results': 'svu_oxygen_concentration_qartod_results',
        'dissolved_oxygen': 'oxygen_concentration_corrected',
        'dissolved_oxygen_qc_executed': 'oxygen_concentration_corrected_qc_executed',
        'dissolved_oxygen_qc_results': 'oxygen_concentration_corrected_qc_results',
        'dissolved_oxygen_qartod_executed': 'oxygen_concentration_corrected_qartod_executed',
        'dissolved_oxygen_qartod_results': 'oxygen_concentration_corrected_qartod_results',
        'int_ctd_pressure': 'sea_water_pressure',
    }
    
    # CGSN Update: made this iterative as above to reprocess limited parameter datasets
    for var in ds.variables:
        if var in rename.keys():
            ds = ds.rename({var: rename.get(var)})

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # add original OOINet variable name as an attribute if renamed
    for key, value in rename.items():
        if value in ds.variables:
            ds[value].attrs['ooinet_variable_name'] = key

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    if burst:   # re-sample the data to a defined time interval using a median average
        # create the burst averaging
        ds['time'] = ds['time'] + np.timedelta64(450, 's')
        burst = ds.resample(time='900s', skipna=True).median(dim='time', keep_attrs=True)
        burst = burst.where(~np.isnan(burst.deployment), drop=True)

        # reset the attributes...which keep_attrs should do...
        burst.attrs = ds.attrs
        for v in burst.variables:
            burst[v].attrs = ds[v].attrs

        # save the newly average data
        ds = burst

    return ds


def dosta_ctdbp_datalogger(ds):
    """
    Takes data from DOSTAs connected to the CTDBP, with the data recorded by
    the datalogger (used by some of the EA moorings) and cleans up the data 
    set to make it more user-friendly. Primary task is renaming the alphabet 
    soup parameter names and dropping some parameters that are of limited or 
    no use/value.

    :param ds: initial dosta data set downloaded from OOI via the M2M system
    :return ds: cleaned up data set
    """
    # drop some of the variables:
    #   dcl_controller_timestamp == time, redundant so can remove
    #   date_time_string == internal_timestamp, redundant so can remove
    drop_list = ['dcl_controller_timestamp', 'date_time_string']
    for var in ds.variables:
        if var in drop_list:
            ds = ds.drop(var)

    # convert the time values from a datetime64[ns] object to a floating point number with the time in seconds
    ds['internal_timestamp'] = ('time', dt64_epoch(ds.internal_timestamp))
    ds['internal_timestamp'].attrs = dict({
        'long_name': 'Internal CTD Clock Time',
        'standard_name': 'time',
        'units': 'seconds since 1970-01-01 00:00:00 0:00',
        'calendar': 'gregorian',
        'comment': ('Comparing the instrument internal clock versus the GPS referenced sampling time will allow for '
                    'calculations of the instrument clock offset and drift. Useful when working with the '
                    'recovered instrument data where no external GPS referenced clock is available.')
    })

    # rename some of the variables for better clarity
    rename = {
        'dosta_analog_tc_oxygen': 'oxygen_concentration',
        'dosta_ln_optode_oxygen': 'oxygen_concentration',
        'dosta_ln_optode_oxygen_qc_executed': 'oxygen_concentration_qc_executed',
        'dosta_ln_optode_oxygen_qc_results': 'oxygen_concentration_qc_results',
        'dosta_ln_optode_oxygen_qartod_executed': 'oxygen_concentration_qartod_executed',
        'dosta_ln_optode_oxygen_qartod_results': 'oxygen_concentration_qartod_results',
        'dissolved_oxygen': 'oxygen_concentration_corrected',
        'dissolved_oxygen_qc_executed': 'oxygen_concentration_corrected_qc_executed',
        'dissolved_oxygen_qc_results': 'oxygen_concentration_corrected_qc_results',
        'dissolved_oxygen_qartod_executed': 'oxygen_concentration_corrected_qartod_executed',
        'dissolved_oxygen_qartod_results': 'oxygen_concentration_corrected_qartod_results',
        'int_ctd_pressure': 'sea_water_pressure',
    }
    for key in rename.keys():
        if key in ds.variables:
            ds = ds.rename({key: rename.get(key)})

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # add original OOINet variable name as an attribute if renamed
    for key, value in rename.items():
        if value in ds.variables:
            ds[value].attrs['ooinet_variable_name'] = key

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    # replace the fill-values (CTD uses a 0 when there is no comms to the DOSTA) with a NaN
    m = ds['oxygen_concentration'] <= 0
    ds['oxygen_concentration'][m] = np.nan
    ds['oxygen_concentration_corrected'][m] = np.nan

    return ds


def dosta_ctdbp_instrument(ds):
    """
    Takes data from DOSTAs connected to the CTDBP, with the data recorded by
    the CTDBP (used by some of the EA moorings) and cleans up the data
    set to make it more user-friendly. Primary task is renaming the alphabet
    soup parameter names and dropping some parameters that are of limited or
    no use/value.

    :param ds: initial dosta data set downloaded from OOI via the M2M system
    :return ds: cleaned up data set
    """
    # drop some of the variables:
    #   ctd_time == time, redundant so can remove
    #   internal_timestamp == time, redundant so can remove
    drop_list = ['internal_timestamp', 'ctd_time']
    for var in ds.variables:
        if var in drop_list:
            ds = ds.drop(var)

    # check for data from a co-located CTD, if not present create the variables using NaN as the fill value
    if 'sea_water_temperature' not in ds.variables:
        ds['sea_water_temperature'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_pressure'] = ('time', ds['deployment'].data * np.nan)

    # rename some variables for better clarity
    rename = {
        'oxygen': 'raw_oxygen_concentration',
        'ctd_tc_oxygen': 'oxygen_concentration',
        'ctd_tc_oxygen_qc_executed': 'oxygen_concentration_qc_executed',
        'ctd_tc_oxygen_qc_results': 'oxygen_concentration_qc_results',
        'ctd_tc_oxygen_qartod_executed': 'oxygen_concentration_qartod_executed',
        'ctd_tc_oxygen_qartod_results': 'oxygen_concentration_qartod_results',
        'dosta_tc_oxygen': 'oxygen_concentration',
        'dosta_tc_oxygen_qc_executed': 'oxygen_concentration_qc_executed',
        'dosta_tc_oxygen_qc_results': 'oxygen_concentration_qc_results',
        'dosta_tc_oxygen_qartod_executed': 'oxygen_concentration_qartod_executed',
        'dosta_tc_oxygen_qartod_results': 'oxygen_concentration_qartod_results',
        'dissolved_oxygen': 'oxygen_concentration_corrected',
        'dissolved_oxygen_qc_executed': 'oxygen_concentration_corrected_qc_executed',
        'dissolved_oxygen_qc_results': 'oxygen_concentration_corrected_qc_results',
        'dissolved_oxygen_qartod_executed': 'oxygen_concentration_corrected_qartod_executed',
        'dissolved_oxygen_qartod_results': 'oxygen_concentration_corrected_qartod_results',
        'int_ctd_pressure': 'sea_water_pressure',
    }
    for key in rename.keys():
        if key in ds.variables:
            ds = ds.rename({key: rename.get(key)})

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # add original OOINet variable name as an attribute if renamed
    for key, value in rename.items():
        if value in ds.variables:
            ds[value].attrs['ooinet_variable_name'] = key

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    # replace the fill-values (CTD uses a 0 when there is no comms to the DOSTA) with a NaN
    m = ds['oxygen_concentration'] <= 0
    ds['oxygen_concentration'][m] = np.nan
    ds['oxygen_concentration_corrected'][m] = np.nan

    return ds


def dosta_cspp(ds):
    """
    Takes DOSTA data recorded by the CSPP loggers used by the Endurance Array
    and cleans up the data set to make it more user-friendly.  Primary task is
    renaming parameters and dropping some that are of limited use. Additionally,
    re-organize some of the variables to permit better assessments of the data.

    :param ds: initial DOSTA data set downloaded from OOI via the M2M system
    :return ds: cleaned up data set
    """
    # drop some of the variables:
    #   suspect_timestamp = not used
    #   internal_timestamp == profiler_timestamp == time, redundant
    #   profiler_timestamp == time, redundant
    #   product_number, these are all 4831s and this is captured in the global attributes
    #   all qc_executed and qc_results variables as these older QC tests are now deprecated
    #       in favor of the QARTOD style tests
    drop_list = ['suspect_timestamp', 'internal_timestamp', 'profiler_timestamp', 'product_number']
    for var in ds.variables:
        if (var in drop_list) or ('_qc_' in var):
            ds = ds.drop_vars(var)

    # rename some variables for better clarity
    rename = {
        'estimated_oxygen_concentration': 'oxygen_concentration',
        'estimated_oxygen_concentration_qartod_executed': 'oxygen_concentration_qartod_executed',
        'estimated_oxygen_concentration_qartod_results': 'oxygen_concentration_qartod_results',
        'estimated_oxygen_saturation': 'oxygen_saturation',
        'dosta_abcdjm_cspp_tc_oxygen': 'svu_oxygen_concentration',
        'dosta_abcdjm_cspp_tc_oxygen_qartod_executed': 'svu_oxygen_concentration_qartod_executed',
        'dosta_abcdjm_cspp_tc_oxygen_qartod_results': 'svu_oxygen_concentration_qartod_results',
        'dissolved_oxygen': 'oxygen_concentration_corrected',
        'dissolved_oxygen_qartod_executed': 'oxygen_concentration_corrected_qartod_executed',
        'dissolved_oxygen_qartod_results': 'oxygen_concentration_corrected_qartod_results',
        'pressure': 'sea_water_pressure',
    }
    ds = ds.rename(rename)

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # add the original variable name as an attribute, if renamed
    for key, value in rename.items():
        ds[value].attrs['ooinet_variable_name'] = key

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    return ds


def dosta_wfp(ds):
    """
    Takes DOSTA data recorded by the WFP loggers used by the Global Arrays
    and cleans up the data set to make it more user-friendly.  Primary task is
    renaming parameters and dropping some that are of limited use. Additionally,
    re-organize some of the variables to permit better assessments of the data.
    :param ds: initial DOSTA data set downloaded from OOI via the M2M system
    :return ds: cleaned up data set
    """
    # drop some of the variables:
    #   suspect_timestamp = not used
    #   internal_timestamp == profiler_timestamp == time, redundant
    #   profiler_timestamp == time, redundant
    #   all qc_executed and qc_results variables as these older QC tests are now deprecated
    #       in favor of the QARTOD style tests
    drop_list = ['suspect_timestamp', 'internal_timestamp', 'profiler_timestamp']
    for var in ds.variables:
        if (var in drop_list) or ('_qc_' in var):
            ds = ds.drop_vars(var)

    # rename some variables for better clarity
    rename = {
        'estimated_oxygen_concentration': 'oxygen_concentration',
        'estimated_oxygen_concentration_qartod_executed': 'oxygen_concentration_qartod_executed',
        'estimated_oxygen_concentration_qartod_results': 'oxygen_concentration_qartod_results',
        'dissolved_oxygen': 'oxygen_concentration_corrected',
        'dissolved_oxygen_qartod_executed': 'oxygen_concentration_corrected_qartod_executed',
        'dissolved_oxygen_qartod_results': 'oxygen_concentration_corrected_qartod_results',
        'int_ctd_pressure': 'sea_water_pressure',
    }
    ds = ds.rename(rename)

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # add the original variable name as an attribute, if renamed
    for key, value in rename.items():
        ds[value].attrs['ooinet_variable_name'] = key

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    return ds


def main(argv=None):
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
    sensor_type = args.sensor_type

    # check if we are specifying a deployment or a specific date and time range
    if not deploy or (start and stop):
        return SyntaxError('You must specify either a deployment number or beginning and end dates of interest.')

    # if we are specifying a deployment number, then get the data from the Gold Copy THREDDS server
    if deploy:
        # download the data for the deployment
        dosta = load_gc_thredds(site, node, sensor, method, stream, ('.*deployment%04d.*DOSTA.*\\.nc$' % deploy))

        # check to see if we downloaded any data
        if not dosta:
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
        dosta = m2m_collect(r, '.*DOSTA.*\\.nc$')

        # check to see if we downloaded any data
        if not dosta:
            exit_text = ('Data unavailable for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                    stream, start, stop))
            raise SystemExit(exit_text)

    if sensor_type not in ['solo', 'ctdbp']:
        exit_text = 'You need to specify the type of DOSTA in order to process: solo or ctdbp'
        raise SystemExit(exit_text)

    # clean-up and reorganize based on the type and data delivery method
    if sensor_type == 'solo':
        if node == 'SP001':
            dosta = dosta_cspp(dosta)
        else:
            dosta = dosta_datalogger(dosta, burst)

    if sensor_type == 'ctdbp':
        if method in ['telemetered', 'recovered_host']:
            dosta = dosta_ctdbp_datalogger(dosta)
        else:
            dosta = dosta_ctdbp_instrument(dosta)

    vocab = get_vocabulary(site, node, sensor)[0]
    dosta = update_dataset(dosta, vocab['maxdepth'])

    # save the data to disk
    out_file = os.path.abspath(args.outfile)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    dosta.to_netcdf(out_file, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)


if __name__ == '__main__':
    main()
