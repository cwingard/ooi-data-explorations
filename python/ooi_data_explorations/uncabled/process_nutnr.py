#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import numpy as np
import os
import pandas as pd

from ooi_data_explorations.common import inputs, m2m_collect, m2m_request, load_gc_thredds, \
    get_vocabulary, update_dataset, ENCODINGS
from ooi_data_explorations.qartod.qc_processing import parse_qc

# load configuration settings
ATTRS = {
    'spectrum_average': {
        'long_name': 'Spectrum Average',
        'comment': 'Average value from the raw spectral measurements.',
        'units': 'counts'
    },
    'nitrate_concentration': {
        'long_name': 'Dissolved Nitrate Concentration',
        'comment': 'Dissolved Nitrate Concentration, uncorrected for temperature and salinity effects.',
        'units': 'umol L-1',
        'data_product_identifier': 'NITRDIS_L1',
    },
    'corrected_nitrate_concentration': {
        'long_name': 'Corrected Dissolved Nitrate Concentration',
        'standard_name': 'mole_concentration_of_nitrate_in_sea_water',
        'comment': 'Temperature and salinity corrected dissolved nitrate concentration.',
        'units': 'umol L-1',
        'data_product_identifier': 'NITRTSC_L2',
        'ancillary_variables': ('sea_water_temperature sea_water_practical_salinity raw_spectral_measurements '
                                'dark_value_used_for_fit')
    },
    'raw_spectral_measurements': {
        'long_name': 'Raw Spectral Measurements',
        'comment': ('The raw absorption spectra is an array of values, measured in counts and produced by the SUNA '
                    'based on the ultraviolet (UV) absorption characteristics of the components of seawater (including '
                    'dissolved nitrate). The raw spectral measurements, between 217 and 240 nm, are used to calculate '
                    'the nitrate concentrations.'),
        'data_product_identifier': 'NITROPT_L0',
        'units': 'counts'
    },
    'wavelength_index': {
        'long_name': 'Wavelength Index',
        'comment': ('Indexing array, with values between 0 and 255, for the raw spectral measurements. Each SUNA '
                    'measures between 190 and 370 nm, approximately every 0.7 nm. The exact wavelengths used are '
                    'specific to each instrument.'),
        # 'units': ''  # deliberately left blank, no units for this variable
    }
}

# Pull in the ts calculation from ion-functions (easier to just add it as
# a function rather than add and import ion-function libraries)
def ts_corrected_nitrate(cal_temp, wl, eno3, eswa, di, dark_value, ctd_t,
                         ctd_sp, data_in, frame_type, wllower=217, wlupper=240):
    """
    Description:

        This Python code is based on Matlab code
        (NUTNR_Example_MATLAB_Code_20140521_ver_1_00.m) that was
        developed by O.E. Kawka (UW/RSN).

        The code below calculates the Dissolved Nitrate Concentration
        with the Sakamoto et. al. (2009) algorithm that uses the observed
        sample salinity and temperature to subtract the bromide component
        of the overall seawater UV absorption spectrum before solving for
        the nitrate concentration.

        The output represents the OOI L2 Dissolved Nitrate Concentration,
        Temperature and Salinity Corrected (NITRTSC).

    Implemented by:

        2014-05-22: Craig Risien. Initial Code
        2014-05-27: Craig Risien. This function now looks for the light vs
                    dark frame measurements and only calculates nitrate
                    concentration based on the light frame measurements.
        2015-04-09: Russell Desiderio. CI is now implementing cal coeffs
                    by tiling in time, requiring coding changes. The
                    tiling includes the wllower and wlupper variables
                    when supplied by CI.

    Usage:

        NO3_conc = ts_corrected_nitrate(cal_temp, wl, eno3, eswa, di,
                                        dark_value, ctd_t, ctd_sp, data_in,
                                        frame_type, wllower, wlupper)

            where

        cal_temp = Calibration water temperature value
        wl = (256,) array of wavelength bins
        eno3 = (256,) array of wavelength-dependent nitrate
                extinction coefficients
        eswa = (256,) array of seawater extinction coefficients
        di = (256,) array of deionized water reference spectrum
        dark_value = (N,) array of dark average scalar value
        ctd_t = (N,) array of water temperature values from
                colocated CTD [deg C].
                (see 1341-00010_Data_Product_Spec_TEMPWAT)
        ctd_sp = (N,) array of practical salinity values from
                colocated CTD [unitless].
                (see 1341-00040_Data_Product_Spec_PRACSAL)
        data_in = (N x 256) array of nitrate measurement values
                from the UV absorption spectrum data product
                (L0 NITROPT) [unitless]
        NO3_conc = L2 Dissolved Nitrate Concentration, Temperature and
                Corrected (NITRTSC) [uM]
        frame_type = (N,) array of Frame type, either a light or dark
                measurement. This function only uses the data from light
                frame measurements.
        wllower = Lower wavelength limit for spectra fit.
                  From DPS: 217 nm (1-cm pathlength probe tip) or
                            220 nm (4-cm pathlength probe tip)
        wlupper = Upper wavelength limit for spectra fit.
                  From DPS: 240 nm (1-cm pathlength probe tip) or
                            245 nm (4-cm pathlength probe tip)
    Notes:

        2015-04-10: R. Desiderio.
            CI has determined that cal coefficients will implemented as time-vectorized
            arguments as inputs to DPAs. This means that all input calibration coefficients
            originally dimensioned as (256,) will now be dimensioned as (N,256), where N is
            the number of data packets.

            This change broke the code ("Blocker Bug #2942") and so necessitated a revision
            of this DPA and its unit test. The useindex construct along with variables WL,
            ENO3, ESWA, and DI were originally set up outside the loop. However, with this CI
            change, it is now possible that the cal coefficients could change inside of the
            cal coeff variable arrays (reflecting data coming from two different instruments).
            I took the conservative approach and moved these calculations inside the loop to
            be calculated for each data packet.

            Fill values on output have been changed to np.nan.

    References:

        OOI (2014). Data Product Specification for NUTNR Data Products.
            Document Control Number 1341-00620.
            https://alfresco.oceanobservatories.org/ (See: Company Home >>
            OOI >> Controlled >> 1000 System Level >>
            1341-00620_Data_Product_Spec_NUTNR_OOI.pdf)
        Johnson, K. S., and L. J. Coletti. 2002. In situ ultraviolet
            spectrophotometry for high resolution and long-term monitoring
            of nitrate, bromide and bisulfide in the ocean. Deep-Sea Res.
            I 49:1291-1305
        Sakamoto, C.M., K.S. Johnson, and L.J. Coletti (2009). Improved
            algorithm for the computation of nitrate concentrations in
            seawater using an in situ ultraviolet spectrophotometer.
            Limnology and Oceanography: Methods 7: 132-143
    """
    n_data_packets = data_in.shape[0]

    # make sure that the dimensionalities of wllower and wlupper are consistent
    # regardless of whether or not they are specified in the argument list.
    if np.isscalar(wllower):
        wllower = np.tile(wllower, n_data_packets)

    if np.isscalar(wlupper):
        wlupper = np.tile(wlupper, n_data_packets)
        
    if np.isscalar(cal_temp):
        cal_temp = np.tile(cal_temp, n_data_packets)
        
    # Make the calibration inputs match the data_in shape
    wl = np.tile(wl, (n_data_packets, 1))
    di = np.tile(di, (n_data_packets, 1))
    eno3 = np.tile(eno3, (n_data_packets, 1))
    eswa = np.tile(eswa, (n_data_packets, 1))

    # coefficients to equation 4 of Sakamoto et al 2009 that give the
    # absorbance of seasalt at 35 salinity versus temperature
    Asak = 1.1500276
    Bsak = 0.02840
    Csak = -0.3101349
    Dsak = 0.001222

    NO3_conc = np.ones(n_data_packets)

    for i in range(0, n_data_packets):

        if frame_type[i] == 'SDB' or frame_type[i] == 'SDF' or frame_type[i] == "NDF":

            ## Ignore and fill dark frame measurements
            #NO3_conc[i] = -9999999.0

            # change this to output nans instead.
            NO3_conc[i] = np.nan

        else:

            # Find wavelength bins that fall between the upper and lower
            # limits for spectra fit
            useindex = np.logical_and(wllower[i] <= wl[i, :], wl[i, :] <= wlupper[i])

            # subset data so that we only use wavelengths between wllower & wlupper
            WL = wl[i, useindex]
            ENO3 = eno3[i, useindex]
            ESWA = eswa[i, useindex]
            DI = np.array(di[i, useindex], dtype='float64')
            SW = np.array(data_in[i, useindex], dtype='float64')

            # correct each SW intensity for dark current
            SWcorr = SW - dark_value[i]

            # calculate absorbance
            Absorbance = np.log10(DI / SWcorr)

            # now estimate molar absorptivity of seasalt at in situ temperature
            # use Satlantic calibration and correct as in Sakamoto et al. 2009.
            SWA_Ext_at_T = (ESWA * ((Asak + Bsak * ctd_t[i]) / (Asak + Bsak * cal_temp[i]))
                            * np.exp(Dsak * (ctd_t[i] - cal_temp[i]) * (WL - 210.0)))

            # absorbance due to seasalt
            A_SWA = ctd_sp[i] * SWA_Ext_at_T
            # subtract seasalt absorbance from measured absorbance
            Acomp = np.array(Absorbance - A_SWA, ndmin=2).T

            # ENO3 plus a linear baseline
            subset_array_size = np.shape(ENO3)
            # for the constant in the linear baseline
            Ones = np.ones((subset_array_size[0],), dtype='float64') / 100
            M = np.vstack((ENO3, Ones, WL / 1000)).T

            # C has NO3, baseline constant, and slope (vs. WL)
            C = np.dot(np.linalg.pinv(M), Acomp)

            NO3_conc[i] = C[0, 0]

    return NO3_conc


def quality_checks(ds):
    """
    Quality assessment of the raw and calculated nitrate concentration data
    using a susbset of the QARTOD flags to indicate the quality. QARTOD
    flags used are:

        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail

    The final flag value represents the worst case assessment of the data quality.

    :param ds: xarray dataset with the raw signal data and the calculated
               seawater pH
    :return qc_flag: array of flag values indicating seawater pH quality
    """
    qc_flag = ds['time'].astype('int32') * 0 + 1   # default flag values, no errors

    # "RMSE: The root-mean-square error parameter from the SUNA V2 can be used to make
    # an estimate of how well the nitrate spectral fit is. This should usually be less than 1E-3. If
    # it is higher, there is spectral shape (likely due to CDOM) that adversely impacts the nitrate
    # estimate." SUNA V2 vendor documentation (Sea-Bird Scientific Document# SUNA180725)
    m = ds.fit_rmse > 0.001  # per the vendor documentation
    qc_flag[m] = 3
    m = ds.fit_rmse > 0.100  # based on experience with the instrument data sets
    qc_flag[m] = 4

    # "Absorption: The data output of the SUNA V2 is the absorption at 350 nm and 254 nm
    # (A350 and A254). These wavelengths are outside the nitrate absorption range and can be
    # used to make an estimate of the impact of CDOM. If absorption is high (>1.3 AU), the
    # SUNA will not be able to collect adequate light to make a measurement." SUNA V2 vendor
    # documentation (Sea-Bird Scientific Document# SUNA180725)
    m254 = ds.absorbance_at_254_nm > 1.3
    qc_flag[m254] = 4
    m350 = ds.absorbance_at_350_nm > 1.3
    qc_flag[m350] = 4

    # test for failed dark value measurements (can't be less than 0)
    m = ds.dark_value_used_for_fit <= 0
    qc_flag[m] = 4

    # test for a blocked absorption channel (or a failed lamp)
    m = ds.spectrum_average < 10000
    qc_flag[m] = 4

    # test for out of range corrected dissolved nitrate readings
    m = (ds.corrected_nitrate_concentration.values < -2.0) | (ds.corrected_nitrate_concentration.values > 300)
    qc_flag[m] = 4

    return qc_flag


def drift_correction(ds, site, node, sensor):
    """
    Apply a drift correction to a processed SUNA dataset.
    
    This function takes in a SUNA dataset which has been
    reprocessed, get the pre-and-post deployment calibrations,
    calculates the differences between the two, and applies
    a linear drift to the temperature-and-salinity corrected
    nitrate concentration. 
    
    :param ds: A SUNA dataset reprocessed using the suna_datalogger
               or suna_instrument functions
    :param site: The site of the associated SUNA dataset
    :param node: The node of the associated SUNA dataset
    :param sensor: The sensor of the associated SUNA dataset
    """
    
    # ----------------------------------------
    # Get the appropriate calibration files and 
    # Coefficients
    # First, grab the deployment of the dataset
    depNum = int(np.unique(ds["deployment"])[0])
    
    # Next, get the deployment start and end times
    deployStart, deployEnd = get_deployment_dates(site, node, sensor, depNum)

    # Use the deployment start and end times to get the deployment specific calibrations
    calInfo = get_calibrations_by_refdes(site, node, sensor, deployStart, deployEnd, to_dataframe=True)
    pre_deploy_cal = calInfo[calInfo["deploymentNumber"] == depNum]

    # Get the UID of the instrument
    uid = pre_deploy_cal["uid"].unique()[0]

    # Now, get all of the deployments for the given instrument
    calInfo = get_calibrations_by_uid(uid, to_dataframe=True)

    # Find the next calibration AFTER the one used for the deployment
    pre_date = pre_deploy_cal["calDate"].unique()[0]
    delta_t = calInfo["calDate"].apply(lambda x: x - pre_date)
    delta_t = delta_t[delta_t.dt.days > 0]
    delta_t = delta_t[delta_t == np.min(delta_t)]

    # Use the delta_t indices to get the post-cruise calibration
    post_deploy_cal = calInfo.loc[delta_t.index]
    
    # ---------------------------------------------------
    # Use the post-cruise DI values to recalculate the 
    # T-S corrected nitrogen
    # Get the calibration values
    cal_temp = np.array(pre_deploy_cal[pre_deploy_cal["calCoef"] == "CC_cal_temp"]["value"].values[0], dtype='float64')
    wl = np.array(pre_deploy_cal[pre_deploy_cal["calCoef"] == "CC_wl"]["value"].values[0], dtype='float64')
    di = np.array(post_deploy_cal[post_deploy_cal["calCoef"] == "CC_di"]["value"].values[0], dtype='float64')
    eno3 = np.array(pre_deploy_cal[pre_deploy_cal["calCoef"] == "CC_eno3"]["value"].values[0], dtype='float64')
    eswa = np.array(pre_deploy_cal[pre_deploy_cal["calCoef"] == "CC_eswa"]["value"].values[0], dtype='float64')

    # Get the spectral input values
    dark_counts = ds['dark_value_used_for_fit'].values
    data_in = ds['raw_spectral_measurements'].values
    frame_type = np.array(len(dark_counts)*['Light'], dtype='str')

    # Get the CTD temperature and salinity values
    ctd_t = ds['sea_water_temperature'].values
    ctd_sp = ds['sea_water_practical_salinity'].values

    # Now run with the post-cal di
    post_cal_nitrate = ts_corrected_nitrate(float(cal_temp), wl, eno3, eswa, di, dark_counts, ctd_t, ctd_sp, data_in, frame_type)
    ds["post_cal_nitrate"] = (["time"], post_cal_nitrate)
    ds["post_cal_nitrate"].attrs = {
        "long_name": "T-S Corrected Dissolved Nitrate Concentration",
        "units": "umol L-1",
        "comment": ("Temperature and salinity corrected dissolved nitrate concentration recalculated "
                    "using the post-cruise DI-water calibration.")
    }
    
    # Get the calibration file names
    pre_deploy_file = pre_deploy_cal["calFile"].unique()[0]
    post_deploy_file = post_deploy_cal["calFile"].unique()[0]
    
    
    # Now, calculate a drift rate based on the offset between the
    # initial T-S corrected nitrogen from the pre-deploy calibration
    # and the post-cruise calibration. This operates on the assumption of
    # a linear drift, thus is LTI
    dno3 = (post_cal_nitrate - ds['corrected_nitrate_concentration']).sel(time=slice(ds["time"].min(),ds["time"].min()+pd.Timedelta('1D'))).mean()
    pre_date = pre_deploy_file.split("__")[-1].split("_")[0]
    post_date = post_deploy_file.split("__")[-1].split("_")[0]
    dt = (pd.to_datetime(post_date) - pd.to_datetime(pre_date)).total_seconds()*1E9
    dno3_dt = dno3 / int(dt)
    delta_t = (ds.time - ds.time.min()).astype('int')

    # Apply the drift correction
    drift_correction = dno3_dt.values*delta_t
    drift_corrected_nitrate = ds["corrected_nitrate_concentration"] + drift_correction
    ds["drift_corrected_nitrate"] = drift_corrected_nitrate
    ds["drift_corrected_nitrate"].attrs = {
        "long_name": "Drift and T-S Corrected Dissolved Nitrate Concentration",
        "units": "umol L-1",
        "comment": ("Temperature and salinity corrected dissolved nitrate concentration with "
                    "linear drift, estimated from the difference between pre-and-post cruise "
                    "DI-water calibrations, removed."),
        "pre_cruise_calibration": pre_deploy_file,
        "post_cruise_calibration": post_deploy_file,
        "dNO3": str(dno3),
        "dt": str(dt)
    }
    
    return ds


def suna_datalogger(ds, burst=False):
    """
    Takes SUNA data recorded by the data loggers used in the CGSN/EA moorings
    and cleans up the data set to make it more user-friendly.  Primary task is
    renaming parameters and dropping some that are of limited use. Additionally,
    re-organize some variables to permit better assessments of the data.

    :param ds: initial nutnr data set downloaded from OOI via the M2M system
    :param burst: resample the data to the defined time interval
    :return ds: cleaned up data set
    """
    # drop some variables:
    #   checksum = used to parse data, is not parsed if the checksum fails so no longer needed
    #   frame_type = remove the dark frames if recorded, then remove
    #   humidity = not measured, no need to include
    ds = ds.reset_coords()
    # CGSN update # 2: now this does a check to the shape, to determine if a 1 or 2-d array
    # For 2-D array, may have to call .value to load into memory if working
    if len(ds["frame_type"].shape) == 1:
        ds['frame_type'] = ds['frame_type'].astype(str)
    else:
        ds['frame_type'] = ('time', [''.join(x.astype(str)) for x in ds.frame_type.data])
    ds = ds.where(ds.frame_type == 'SLF', drop=True)  # remove the dark frames
    ds = ds.drop_vars(['checksum', 'frame_type', 'humidity'])

    # convert the instrument date and time values to a floating point number with the time in seconds, and then
    # update the internal_timestamp with the values as it was not set correctly during parsing and defaults to 1900.
    date_of_sample = ds.date_of_sample.values.astype(str)
    ddate = [datetime.datetime.strptime(dd[:7], '%Y%j') for dd in date_of_sample]
    dhour = [datetime.timedelta(hours=dh) for dh in ds.time_of_sample.values]
    internal_timestamp = np.array([np.datetime64(ddate[i] + dhour[i]) for i in range(len(ddate))])
    ds['internal_timestamp'] = ('time', internal_timestamp.astype(float) / 10.0 ** 6)
    ds['internal_timestamp'].attrs = dict({
        'long_name': 'Internal NUTNR Clock Time',
        'standard_name': 'time',
        'units': 'seconds since 1900-01-01T00:00:00.000Z',
        'calendar': 'gregorian',
        'comment': ('Comparing the instrument internal clock versus the GPS referenced sampling time will allow for '
                    'calculations of the instrument clock offset and drift. Useful when working with the '
                    'recovered instrument data where no external GPS referenced clock is available.')
    })
    ds = ds.drop_vars(['date_of_sample', 'time_of_sample'])

    # check for data from a co-located CTD, if not present add with appropriate attributes
    if 'sea_water_temperature' not in ds.variables:
        ds['sea_water_temperature'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_temperature'].attrs = {
            'comment': ('Normally this would be sea water temperature data from a co-located CTD. However, data from ' +
                        'that sensor is unavailable. This value has been filled with NaNs to preserve the structure ' +
                        'of the data set.'),
            'data_product_identifier': 'TEMPWAT_L1',
            'long_name': 'Sea Water Temperature',
            'standard_name': 'sea_water_temperature',
            'units': 'degree_Celsius'
        }

        ds['sea_water_practical_salinity'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_practical_salinity'].attrs = {
            'long_name': 'Sea Water Practical Salinity',
            'standard_name': 'sea_water_practical_salinity',
            'units': '1',
            'comment': ('Normally this would be seawater salinity data from a co-located CTD. However, data from ' +
                        'that sensor is unavailable. This value has been filled with NaNs to preserve the structure ' +
                        'of the data set.'),
            'data_product_identifier': 'PRACSAL_L2'
        }

    # rename some variables for better clarity
    rename = {
        'nutnr_absorbance_at_254_nm': 'absorbance_at_254_nm',
        'nutnr_absorbance_at_350_nm': 'absorbance_at_350_nm',
        'nutnr_bromide_trace': 'bromide_trace',
        'nutnr_current_main': 'current_main',
        'nutnr_dark_value_used_for_fit': 'dark_value_used_for_fit',
        'nutnr_fit_base_1': 'fit_base_1',
        'nutnr_fit_base_2': 'fit_base_2',
        'nutnr_fit_rmse': 'fit_rmse',
        'nutnr_integration_time_factor': 'integration_time_factor',
        'nutnr_nitrogen_in_nitrate': 'nitrogen_in_nitrate',
        'nutnr_spectrum_average': 'spectrum_average',
        'nutnr_voltage_int': 'voltage_instrument',
        'temp_spectrometer': 'temperature_spectrometer',
        'temp_lamp': 'temperature_lamp',
        'temp_interior': 'temperature_interior',
        'spectral_channels': 'raw_spectral_measurements',
        'salinity_corrected_nitrate': 'corrected_nitrate_concentration',
        'salinity_corrected_nitrate_qc_results': 'corrected_nitrate_concentration_qc_results',
        'salinity_corrected_nitrate_qc_executed': 'corrected_nitrate_concentration_qc_executed',
        'salinity_corrected_nitrate_qartod_results': 'corrected_nitrate_concentration_qartod_results',
        'salinity_corrected_nitrate_qartod_executed': 'corrected_nitrate_concentration_qartod_executed',
        'wavelength': 'wavelength_index'
    }
    for key, value in rename.items():
        if key in ds.variables:
            ds = ds.rename({key: value})
            ds[value].attrs['ooinet_variable_name'] = key

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # address incorrectly set units and variable types
    ds['dark_value_used_for_fit'].attrs['units'] = 'counts'
    # CGSN update # 2: now this does a check to the shape, to determine if a 1 or 2-d array
    # For 2-D array, may have to call .value to load into memory if working
    if len(ds['serial_number'].shape) == 1:
        ds['serial_number'] = ds['serial_number'].astype(int)
    else:
        ds['serial_number'] = ('time', [int(''.join(x.astype(str))) for x in ds.serial_number.data])
        ds['serial_number'].attrs = dict({
            'long_name': 'Serial Number',
            # 'units': '', # deliberately left blank, unit-less value
            'comment': 'Instrument serial number',
        })

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    # test the data quality using additional instrument variables
    ds['nitrate_sensor_quality_flag'] = quality_checks(ds)

    if burst:   # re-sample the data to a defined time interval using a median average
        # create the burst averaging
        ds['time'] = ds['time'] + np.timedelta64(450, 's')
        burst = ds.resample(time='900s', skipna=True).median(dim='time', keep_attrs=True)
        burst = burst.where(~np.isnan(burst.deployment), drop=True)

        # reset the attributes...which keep_attrs should do...
        burst.attrs = ds.attrs
        for v in burst.variables:
            burst[v].attrs = ds[v].attrs

        # save the newly averaged data
        ds = burst.copy()

        # and reset some data types to integers
        data_types = ['deployment', 'spectrum_average', 'serial_number', 'dark_value_used_for_fit',
                      'raw_spectral_measurements', 'corrected_nitrate_concentration_qc_executed',
                      'corrected_nitrate_concentration_qc_results', 'corrected_nitrate_concentration_qartod_results',
                      'nitrate_sensor_quality_flag']
        for v in data_types:
            ds[v] = ds[v].round().astype('int32')

    return ds


def suna_instrument(ds, burst=True):
    """
    Takes SUNA data recorded internally by the Sea-Bird SUNA dissolved nitrate
    sensor and cleans up the data set to make it more user-friendly.  Primary
    task is renaming parameters and dropping some that are of limited use.
    Additionally, re-organize some variables to permit better assessments of
    the data.

    :param ds: initial nutnr data set downloaded from OOI via the M2M system
    :param burst: resample the data to the defined time interval
    :return ds: cleaned up data set
    """
    # drop some variables:
    #   checksum = used to parse data, is not parsed if the checksum fails so no longer needed
    #   frame_type = remove the dark frames if recorded, then remove
    #   humidity = not measured, no need to include
    #   internal_timestamp == time, redundant so can remove
    #   date_of_sample = used to construct the internal_timestamp
    #   time_of_sample = used to construct the internal_timestamp
    ds = ds.reset_coords()
    # CGSN update: switched to simple type conversion - handles byte type arrays
    ds['frame_type'] = ('time', [''.join(x.astype(str)) for x in ds.frame_type.data])
    ds = ds.where(ds.frame_type == 'SLF', drop=True)  # remove the dark frames
    ds = ds.drop_vars(['checksum', 'frame_type', 'humidity', 'date_of_sample', 'time_of_sample'])

    # check for data from a co-located CTD, if not present add with appropriate attributes
    if 'sea_water_temperature' not in ds.variables:
        ds['sea_water_temperature'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_temperature'].attrs = {
            'comment': ('Normally this would be sea water temperature data from a co-located CTD. However, data from ' +
                        'that sensor is unavailable. This value has been filled with NaNs to preserve the structure ' +
                        'of the data set.'),
            'data_product_identifier': 'TEMPWAT_L1',
            'long_name': 'Sea Water Temperature',
            'standard_name': 'sea_water_temperature',
            'units': 'degree_Celsius'
        }

        ds['sea_water_practical_salinity'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_practical_salinity'].attrs = {
            'long_name': 'Sea Water Practical Salinity',
            'standard_name': 'sea_water_practical_salinity',
            'units': '1',
            'comment': ('Normally this would be seawater salinity data from a co-located CTD. However, data from ' +
                        'that sensor is unavailable. This value has been filled with NaNs to preserve the structure ' +
                        'of the data set.'),
            'data_product_identifier': 'PRACSAL_L2'
        }

    # rename some variables for better clarity
    rename = {
        'nutnr_absorbance_at_254_nm': 'absorbance_at_254_nm',
        'nutnr_absorbance_at_350_nm': 'absorbance_at_350_nm',
        'nutnr_bromide_trace': 'bromide_trace',
        'nutnr_current_main': 'current_main',
        'nutnr_dark_value_used_for_fit': 'dark_value_used_for_fit',
        'nutnr_fit_base_1': 'fit_base_1',
        'nutnr_fit_base_2': 'fit_base_2',
        'nutnr_fit_rmse': 'fit_rmse',
        'nutnr_integration_time_factor': 'integration_time_factor',
        'nutnr_nitrogen_in_nitrate': 'nitrogen_in_nitrate',
        'nutnr_spectrum_average': 'spectrum_average',
        'nutnr_voltage_int': 'voltage_instrument',
        'temp_spectrometer': 'temperature_spectrometer',
        'temp_lamp': 'temperature_lamp',
        'temp_interior': 'temperature_interior',
        'spectral_channels': 'raw_spectral_measurements',
        'salinity_corrected_nitrate': 'corrected_nitrate_concentration',
        'salinity_corrected_nitrate_qc_results': 'corrected_nitrate_concentration_qc_results',
        'salinity_corrected_nitrate_qc_executed': 'corrected_nitrate_concentration_qc_executed',
        'salinity_corrected_nitrate_qartod_results': 'corrected_nitrate_concentration_qartod_results',
        'salinity_corrected_nitrate_qartod_executed': 'corrected_nitrate_concentration_qartod_executed',
        'wavelength': 'wavelength_index'
    }
    for key, value in rename.items():
        if key in ds.variables:
            ds = ds.rename({key: value})
            ds[value].attrs['ooinet_variable_name'] = key

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # address incorrectly set units and variable types
    ds['dark_value_used_for_fit'].attrs['units'] = 'counts'
    # CGSN update # 2: now this does a check to the shape, to determine if a 1 or 2-d array
    # For 2-D array, may have to call .value to load into memory if working
    if len(ds['serial_number'].shape) == 1:
        ds['serial_number'] = ds['serial_number'].astype(int)
    else:
        ds['serial_number'] = ('time', [int(''.join(x.astype(str))) for x in ds.serial_number.data])
        ds['serial_number'].attrs = dict({
            'long_name': 'Serial Number',
            # 'units': '', # deliberately left blank, unit-less value
            'comment': 'Instrument serial number',
        })

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    # test the data quality using additional instrument variables
    ds['nitrate_sensor_quality_flag'] = quality_checks(ds)

    if burst:   # re-sample the data to a defined time interval using a median average
        # create the burst averaging
        ds['time'] = ds['time'] + np.timedelta64(450, 's')
        burst = ds.resample(time='900s', skipna=True).median(dim='time', keep_attrs=True)
        burst = burst.where(~np.isnan(burst.deployment), drop=True)

        # reset the attributes...which keep_attrs should do...
        burst.attrs = ds.attrs
        for v in burst.variables:
            burst[v].attrs = ds[v].attrs

        # save the newly averaged data
        ds = burst.copy()

        # and reset some data types to integers
        data_types = ['deployment', 'spectrum_average', 'serial_number', 'dark_value_used_for_fit',
                      'raw_spectral_measurements', 'corrected_nitrate_concentration_qc_executed',
                      'corrected_nitrate_concentration_qc_results', 'corrected_nitrate_concentration_qartod_results',
                      'nitrate_sensor_quality_flag']
        for v in data_types:
            ds[v] = ds[v].round().astype('int32')

    return ds


def suna_cspp(ds):
    """
    Takes SUNA data recorded by the CSPP loggers used by the Endurance Array
    and cleans up the data set to make it more user-friendly.  Primary task is
    renaming parameters and dropping some that are of limited use. Additionally,
    re-organize some variables to permit better assessments of the data.

    :param ds: initial NUTNR data set downloaded from OOI as NetCDF file
    :return ds: cleaned up data set
    """
    # re-join the frame type into a single string rather than an array
    ds = ds.reset_coords()
    ds['frame_type'] = ('time', [''.join(x.astype(str)) for x in ds.frame_type.data])
    ds = ds.where(ds.frame_type == 'SLB', drop=True)  # keep the light frames, dropping the dark frames

    # drop some variables:
    #   suspect_timestamp = not used
    #   internal_timestamp = not accurate, don't use
    #   day_of_year = used to construct the internal_timestamp, not accurate
    #   time_of_sample = used to construct the internal_timestamp, not accurate
    #   year = used to construct the internal_timestamp, not accurate
    #   ctd_time_uint32 = filled with 0's, dropping
    #   profiler_timestamp == time, redundant
    #   frame_type = no longer needed after using to filter above
    #   nutnr_nitrogen_in_nitrate_qc_* = incorrectly applied QC tests
    #   sea_water_*_qc_* = surprisingly, gross range tests against fill values fail....
    drop_list = ['suspect_timestamp', 'internal_timestamp', 'day_of_year', 'time_of_sample', 'year',
                 'ctd_time_uint32', 'profiler_timestamp', 'frame_type',
                 'nutnr_nitrogen_in_nitrate_qc_executed', 'nutnr_nitrogen_in_nitrate_qc_results',
                 'sea_water_pressure_qc_executed', 'sea_water_pressure_qc_results',
                 'sea_water_temperature_qc_executed', 'sea_water_temperature_qc_results',
                 'sea_water_practical_salinity_qc_executed', 'sea_water_practical_salinity_qc_results']
    for var in ds.variables:
        if var in drop_list:
            ds = ds.drop(var)

    # rename some variables for better clarity
    rename = {
        'nutnr_absorbance_at_254_nm': 'absorbance_at_254_nm',
        'nutnr_absorbance_at_350_nm': 'absorbance_at_350_nm',
        'nutnr_bromide_trace': 'bromide_trace',
        'nutnr_current_main': 'current_main',
        'nutnr_dark_value_used_for_fit': 'dark_value_used_for_fit',
        'nutnr_fit_base_1': 'fit_base_1',
        'nutnr_fit_base_2': 'fit_base_2',
        'nutnr_fit_rmse': 'fit_rmse',
        'nutnr_integration_time_factor': 'integration_time_factor',
        'nutnr_nitrogen_in_nitrate': 'nitrogen_in_nitrate',
        'nutnr_spectrum_average': 'spectrum_average',
        'nutnr_voltage_int': 'voltage_instrument',
        'temp_spectrometer': 'temperature_spectrometer',
        'temp_lamp': 'temperature_lamp',
        'temp_interior': 'temperature_interior',
        'spectral_channels': 'raw_spectral_measurements',
        'salinity_corrected_nitrate': 'corrected_nitrate_concentration',
        'salinity_corrected_nitrate_qc_results': 'corrected_nitrate_concentration_qc_results',
        'salinity_corrected_nitrate_qc_executed': 'corrected_nitrate_concentration_qc_executed',
        'wavelength': 'wavelength_index'
    }
    for key, value in rename.items():
        if key in ds.variables:
            ds = ds.rename({key: value})
            ds[value].attrs['ooinet_variable_name'] = key

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # properly assign the co-located CTD data to the correct variable names
    ds['sea_water_pressure'] = ds['int_ctd_pressure']
    ds['sea_water_temperature'] = ds['ctdpf_j_cspp_instrument_recovered-sea_water_temperature']
    ds['sea_water_practical_salinity'] = ds['ctdpf_j_cspp_instrument_recovered-sea_water_practical_salinity']
    ds = ds.drop(['int_ctd_pressure', 'ctdpf_j_cspp_instrument_recovered-sea_water_practical_salinity',
                  'ctdpf_j_cspp_instrument_recovered-sea_water_temperature'])

    # address incorrectly set units and variable types
    ds['dark_value_used_for_fit'].attrs['units'] = 'counts'

    # parse the OOI QC variables and add QARTOD style QC summary flags to the data, converting the
    # bitmap represented flags into an integer value representing pass == 1, suspect or of high
    # interest == 3, and fail == 4.
    ds = parse_qc(ds)

    # test the data quality using additional instrument variables
    ds['nitrate_sensor_quality_flag'] = quality_checks(ds)

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

    if stream not in ['suna_dcl_recovered', 'suna_instrument_recovered']:
        raise SystemExit('Data from the Satlantic ISUS dissolved nitrate sensor will not be supported at this time.')

    # check if we are specifying a deployment or a specific date and time range
    if not deploy or (start and stop):
        return SyntaxError('You must specify either a deployment number or beginning and end dates of interest.')

    # if we are specifying a deployment number, then get the data from the Gold Copy THREDDS server
    if deploy:
        # download the data for the deployment
        nutnr = load_gc_thredds(site, node, sensor, method, stream, ('.*deployment%04d.*NUTNR.*\\.nc$' % deploy))

        # check to see if we downloaded any data
        if not nutnr:
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
        nutnr = m2m_collect(r, '.*NUTNR.*\\.nc$')

        # check to see if we downloaded any data
        if not nutnr:
            exit_text = ('Data unavailable for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                    stream, start, stop))
            raise SystemExit(exit_text)

    # clean-up and reorganize
    if node == 'SP001':
        # this NUTNR is part of a CSPP
        nutnr = suna_cspp(nutnr)
    else:
        if method in ['telemetered', 'recovered_host']:
            nutnr = suna_datalogger(nutnr, burst)
        else:
            nutnr = suna_instrument(nutnr, burst)

    vocab = get_vocabulary(site, node, sensor)[0]
    nutnr = update_dataset(nutnr, vocab['maxdepth'])

    # save the data to disk
    out_file = os.path.abspath(args.outfile)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    nutnr.to_netcdf(out_file, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)


if __name__ == '__main__':
    main()
