#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import re

from copy import copy
from functools import partial
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from ooi_data_explorations.calibrations import Coefficients, compare_names
from pyseas.data.opt_functions_tscor import tscor


class Calibrations(Coefficients):
    def __init__(self, coeff_file):
        """
        Loads the OPTAA factory calibration coefficients for a unit. Values
        come from either a serialized object created per instrument and
        deployment (calibration coefficients do not change in the middle of a
        deployment), or from the calibration data available from the OOI M2M
        API.
        """
        # assign the inputs
        Coefficients.__init__(self, coeff_file)

    def parse_m2m_cals(self, serial_number, cals, cal_idx):
        """
        Parse the calibration data from the M2M object store. The calibration
        data is stored as an unsorted list of dictionaries. The cal_idx is
        used to identify the calibration data of interest in each dictionary.

        :param serial_number: instrument serial number
        :param cals: list of calibration dictionaries
        :param cal_idx: dictionary of calibration indices
        :return: dictionary of calibration coefficients
        """
        # create the device file dictionary and assign values
        coeffs = {}
        cal_name = None
        # parse the calibration data
        for cal in cals:
            # beam attenuation and absorption channel clear water offsets
            if cal['name'] == 'CC_acwo':
                coeffs['a_offsets'] = np.array(cal['calData'][cal_idx['CC_acwo']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_acwo']]['dataSource'])
            if cal['name'] == 'CC_ccwo':
                coeffs['c_offsets'] = np.array(cal['calData'][cal_idx['CC_ccwo']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_ccwo']]['dataSource'])
            # beam attenuation and absorption channel wavelengths
            if cal['name'] == 'CC_awlngth':
                coeffs['a_wavelengths'] = np.array(cal['calData'][cal_idx['CC_awlngth']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_awlngth']]['dataSource'])
            if cal['name'] == 'CC_cwlngth':
                coeffs['c_wavelengths'] = np.array(cal['calData'][cal_idx['CC_cwlngth']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_cwlngth']]['dataSource'])
            # internal temperature compensation values
            if cal['name'] == 'CC_tbins':
                coeffs['temp_bins'] = np.array(cal['calData'][cal_idx['CC_tbins']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_tbins']]['dataSource'])
            # temperature of calibration water
            if cal['name'] == 'CC_tcal':
                coeffs['temp_calibration'] = cal['calData'][cal_idx['CC_tcal']]['value']
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_tcal']]['dataSource'])
            # temperature compensation values as f(wavelength, temperature) for the attenuation and absorption channels
            if cal['name'] == 'CC_tcarray':
                coeffs['tc_array'] = np.array(cal['calData'][cal_idx['CC_tcarray']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_tcarray']]['dataSource'])
            if cal['name'] == 'CC_taarray':
                coeffs['ta_array'] = np.array(cal['calData'][cal_idx['CC_taarray']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_taarray']]['dataSource'])

        # calibration data file name
        coeffs['source_file'] = cal_name
        # number of wavelengths
        coeffs['num_wavelengths'] = len(coeffs['a_wavelengths'])
        # number of internal temperature compensation bins
        coeffs['num_temp_bins'] = len(coeffs['temp_bins'])
        # pressure coefficients, set to 0 since not included in the CI csv files
        coeffs['pressure_coeff'] = [0, 0]

        # determine the grating index
        awlngths = copy(coeffs['a_wavelengths'])
        awlngths[(awlngths < 545) | (awlngths > 605)] = np.nan
        cwlngths = copy(coeffs['c_wavelengths'])
        cwlngths[(cwlngths < 545) | (cwlngths > 605)] = np.nan
        grate_index = np.nanargmin(np.diff(awlngths) + np.diff(cwlngths))
        coeffs['grate_index'] = grate_index

        # serial number, stripping off all but the numbers
        coeffs['serial_number'] = int(re.sub('[^0-9]', '', serial_number))

        # save the resulting dictionary
        self.coeffs = coeffs


def convert_raw(ref, sig, tintrn, offset, tarray, tbins):
    """
    Convert the raw reference and signal measurements to scientific units. Uses
    a simplified version of the opt_pd_calc function from the pyseas library
    (a fork of the OOI ion_functions code converted to Python 3) to take
    advantage of numpy arrays and the ability to vectorize some of the
    calculations.

    :param ref: raw reference light measurements (OPTCREF_L0 or OPTAREF_L0, as
        appropriate) [counts]
    :param sig: raw signal light measurements (OPTCSIG_L0 or OPTASIG_L0, as
            appropriate) [counts]
    :param tintrn: internal instrument temperature [deg_C]
    :param offset: 'a' or 'c' (as appropriate) clear water offsets from the
        AC-S device file [m-1]
    :param tarray: instrument, wavelength and channel ('c' or 'a') specific
        internal temperature calibration correction coefficients from AC-S
        device file [m-1]
    :param tbins: instrument specific internal temperature calibration bin
        values from AC-S device file [deg_C]
    :return: uncorrected beam attenuation/optical absorption coefficients [m-1]
    """
    # create a linear temperature correction factor based on the internal instrument temperature by interpolating
    # between the temperature bins in the calibration file that bracket the internal temperature.
    if tintrn < tbins[0] or tintrn > tbins[-1]:
        # if the internal temperature is out of range, return NaNs. this can happen if the internal temperature sensor
        # is not working properly (rare, but it has happened).
        pg = np.ones(sig.shape) * np.nan
        return pg

    t0 = tbins[tbins - tintrn < 0][-1]  # set first bracketing temperature
    t1 = tbins[tintrn - tbins < 0][0]   # set second bracketing temperature

    # use the temperature bins to select the calibration coefficients bracketing the internal
    tbins = list(tbins)
    dt0 = tarray[:, tbins.index(t0)]
    dt1 = tarray[:, tbins.index(t1)]

    # Calculate the linear temperature correction.
    tcorr = dt0 + ((tintrn - t0) / (t1 - t0)) * (dt1 - dt0)

    # convert the raw data to the uncorrected spectra
    pg = (offset - (1. / 0.25) * np.log(sig / ref)) - tcorr
    return pg


def holo_grater(wlngths, spectra, index):
    """
    Derived from the Matlab HoloGrater function in Jesse Bausell's
    acsPROCESS_INTERACTIVE toolbox (link below) used in preparing
    AC-S data for NASA's SeaBASS submission process. From the
    original source:

    "This function performs the holographic grating correction for raw
    AC-S spectra. For each individual spectrum it calculates expected
    absorption/attenuation at the lowest wavelength of the second grating
    (upper wavelengths) using matlab's spline function. It then subtracts
    this value from the observed absorption/attenuation creating an offset."

    This function utilizes the SciPy CubicSpline function to accomplish the
    same correction.

    For original code see https://github.com/JesseBausell/acsPROCESS_INTERACTIVE

    :param wlngths: absorption or attenuation channel wavelengths [nm]
    :param spectra: absorption or attenuation spectra [m-1]
    :param index: index of the second holographic grating
    :return: corrected spectra and offset
    """
    # Interpolate between holographic gratings and calculate the offset
    spl = CubicSpline(wlngths[index - 2:index + 1], spectra[index - 2:index + 1])
    interpolation = spl(wlngths[index + 1], extrapolate=True)

    # calculate the offset as the difference between the observed and expected absorption/attenuation
    offset = interpolation - spectra[index + 1]

    # use the offset to correct the second holographic grating
    spectra[index + 1:] = spectra[index + 1:] + offset

    return spectra, offset


def pg_calc(reference, signal, internal_temperature, offset, tarray, tbins, grating, wavelengths, pre_cal=None):
    """
    Combines the convert_raw and holo_grater functions to calculate the L1 data
    products for the AC-S (the uncorrected optical absorption and the beam
    attenuation). This function is intended to be wrapped by the apply_dev
    function to facilitate multiprocessing of the potentially large data arrays
    common to the AC-S.

    :param reference: raw reference light measurements
    :param signal: raw signal light measurements
    :param internal_temperature: internal instrument temperature converted from
        raw counts to degrees Celsius
    :param offset: 'a' or 'c' (as appropriate) clear water offsets from the
        AC-S device file
    :param tarray: instrument, wavelength and channel ('c' or 'a') specific
        internal temperature calibration correction coefficients from AC-S
        device file
    :param tbins: instrument specific internal temperature calibration bin
        values from AC-S device file
    :param grating: index of the for the start of the second half of the
        filter sets
    :param wavelengths: instrument, wavelength and channel ('c' or 'a') specific
        wavelengths from the AC-S device file
    :param pre_cal: pure water calibration offsets for the AC-S from
        pre-deployment calibrations device file
    :return: converted optical absorption or beam attenuation measurements
        along with the offset correction for the holographic grating.
    """
    # convert the raw measurements to uncorrected absorption/attenuation values
    pg = convert_raw(reference, signal, internal_temperature, offset, tarray, tbins)

    # if pre-deployment, pure-water calibration offsets are provided, subtract them from the
    # uncorrected absorption/attenuation values
    if pre_cal is not None:
        pg = pg - pre_cal    # subtract the pure water offset

    # if the grating index is set, correct for the often observed jump at the mid-point of the spectra
    if grating and np.all(~np.isnan(pg)):
        pg, jump = holo_grater(wavelengths, pg, grating)
    else:
        jump = np.nan

    return pg, jump


def apply_dev(optaa, coeffs, pre_cal=None):
    """
    Processes the raw data contained in the optaa dictionary and applies the
    factory calibration coefficients contained in the coeffs dictionary to
    convert the data into initial science units. Processing includes correcting
    for the holographic grating offset common to AC-S instruments.

    :param optaa: xarray dataset with the raw absorption and beam attenuation
        measurements.
    :param coeffs: Factory calibration coefficients in a dictionary structure
    :param pre_cal: optional, pre-deployment, pure water calibration offsets
        for the AC-S.

    :return optaa: xarray dataset with the raw absorption and beam attenuation
        measurements converted into particulate and beam attenuation values
        with the factory pure water calibration values subtracted.
    """
    # pull out the relevant raw data parameters
    nrows = optaa['a_reference'].shape[0]
    a_ref = optaa['a_reference'].values
    a_sig = optaa['a_signal'].values
    c_ref = optaa['c_reference'].values
    c_sig = optaa['c_signal'].values
    temp_internal = optaa['internal_temp'].values

    # replace any 0's in the signal or reference with 1's to avoid divide by zero errors
    a_sig[a_sig == 0] = 1
    a_ref[a_ref == 0] = 1
    c_sig[c_sig == 0] = 1
    c_ref[c_ref == 0] = 1

    # if pre-deployment, pure-water calibration offsets are provided, create variables to hold them for subsequent
    # subtraction from the uncorrected absorption/attenuation values
    if pre_cal is None:
        apre = None
        cpre = None
    else:
        apre = pre_cal['a']
        cpre = pre_cal['c']

    # create a set of partial functions for the subsequent list comprehension (maps static elements to iterables)
    apg_calc = partial(pg_calc, offset=coeffs['a_offsets'], tarray=coeffs['ta_array'],
                       tbins=coeffs['temp_bins'], grating=coeffs['grate_index'],
                       wavelengths=coeffs['a_wavelengths'], pre_cal=apre)
    cpg_calc = partial(pg_calc, offset=coeffs['c_offsets'], tarray=coeffs['tc_array'],
                       tbins=coeffs['temp_bins'], grating=coeffs['grate_index'],
                       wavelengths=coeffs['c_wavelengths'], pre_cal=cpre)

    # apply the partial functions to the data arrays, calculating the L1 data products
    apg = [apg_calc(a_ref[i, :], a_sig[i, :], temp_internal[i]) for i in tqdm(range(nrows),
                                                                              desc='Converting absorption data ...')]
    cpg = [cpg_calc(c_ref[i, :], c_sig[i, :], temp_internal[i]) for i in tqdm(range(nrows),
                                                                              desc='Converting attenuation data ...')]

    # create data arrays of the L1 data products
    apg, a_jumps = zip(*[row for row in apg])
    apg = np.array(apg)
    m = ~np.isfinite(apg)
    apg[m] = np.nan
    a_jumps = np.array(a_jumps)

    cpg, c_jumps = zip(*[row for row in cpg])
    cpg = np.array(cpg)
    m = ~np.isfinite(cpg)
    cpg[m] = np.nan
    c_jumps = np.array(c_jumps)

    # return the L1 data with the factory calibrations applied and the spectral jump corrected (if available)
    optaa['apg'] = (('time', 'wavelength_number'), apg)
    optaa['cpg'] = (('time', 'wavelength_number'), cpg)
    optaa['a_jump_offsets'] = ('time', a_jumps)
    optaa['c_jump_offsets'] = ('time', c_jumps)
    return optaa


def tempsal_corr(channel, pg, wlngth, tcal, temperature, salinity):
    """
    Apply temperature and salinity corrections to the converted absorption
    and attenuation data. Uses a simplified version of the opt_tempsal_corr
    function from the pyseas library (fork of the OOI ion_functions code
    converted to Python 3) to take advantage of numpy arrays and the
    ability to "vectorize" some of the calculations.

    :param channel: string ('a' or 'c') indicating either the absorption or
        attenuation channel is being corrected
    :param pg: array of converted absorption or attenuation data
    :param wlngth: absorption or attenuation channel wavelengths from the
        calibration coefficients
    :param tcal: temperature of the pure water used in the calibrations
    :param temperature: in-situ temperature, ideally from a co-located CTD
    :param salinity: in-situ salinity, ideally from a co-located CTD
    :return: temperature and salinity corrected data
    """
    # create the temperature and salinity correction arrays for each wavelength
    cor_coeffs = np.array([tscor[ii] for ii in wlngth])
    nrows = len(temperature)

    temp_corr = np.tile(cor_coeffs[:, 0], [nrows, 1])
    saln_c_corr = np.tile(cor_coeffs[:, 1], [nrows, 1])
    saln_a_corr = np.tile(cor_coeffs[:, 2], [nrows, 1])

    delta_temp = np.atleast_2d(temperature - tcal).T
    salinity = np.atleast_2d(salinity).T

    if channel == 'a':
        pg_ts = pg - delta_temp * temp_corr - salinity * saln_a_corr
    elif channel == 'c':
        pg_ts = pg - delta_temp * temp_corr - salinity * saln_c_corr
    else:
        raise ValueError('Channel must be either "a" or "c"')

    return pg_ts


def apply_tscorr(optaa, coeffs, temperature, salinity):
    """
    Corrects the absorption and beam attenuation data for the absorption
    of seawater as a function of the seawater temperature and salinity (the
    calibration blanking offsets are determined using pure water.)

    If inputs temperature or salinity are not supplied as calling arguments,
    or all of the temperature or salinity values are NaN, then the following
    default values are used.

        temperature: temperature values recorded by the AC-S's external
            thermistor (note, this would not be valid for an AC-S on a
            profiling platform)
        salinity: 34.0 psu

    Otherwise, each of the arguments for temp and salinity should be either a
    scalar, or a 1D array or a row or column vector with the same number of time
    points as 'a' and 'c'.

    :param optaa: xarray dataset with the raw absorption and attenuation data
        converted to absorption and attenuation coefficients
    :param coeffs: Factory calibration coefficients in a dictionary structure
    :param temperature: In-situ seawater temperature, from a co-located CTD
    :param salinity: In-situ seawater salinity, from a co-located CTD

    :return optaa: xarray dataset with the temperature and salinity corrected
        absorbance and attenuation data arrays added.
    """
    # check the temperature and salinity inputs. If they are not supplied, use the
    # external thermistor temperature and a salinity of 34.0 psu
    if temperature is None:
        temperature = optaa['external_temp']
    if salinity is None:
        salinity = np.ones_like(temperature) * 34.0

    # additionally check if all the temperature and salinity values are NaNs. If they are,
    # then use the external thermistor temperature and a salinity of 34.0 psu (will occur
    # if the CTD is not connected).
    if np.all(np.isnan(temperature)):
        temperature = optaa['external_temp']
    if np.all(np.isnan(salinity)):
        salinity = np.ones_like(temperature) * 34.0

    # test if the temperature and salinity are the same size as the absorption and attenuation
    # data. If they are not, then they should be a scalar value, and we can tile them to the
    # correct size.
    if temperature.size != optaa['time'].size:
        temperature = np.tile(temperature, optaa['time'].size)
    if salinity.size != optaa['time'].size:
        salinity = np.tile(salinity, optaa['time'].size)

    # apply the temperature and salinity corrections
    optaa['apg_ts'] = tempsal_corr('a', optaa['apg'], coeffs['a_wavelengths'], coeffs['temp_calibration'],
                                   temperature, salinity)
    optaa['cpg_ts'] = tempsal_corr('c', optaa['cpg'], coeffs['c_wavelengths'], coeffs['temp_calibration'],
                                   temperature, salinity)

    return optaa


def apply_scatcorr(optaa, coeffs):
    """
    Correct the absorbance data for scattering using Method 1, with the
    wavelength closest to 715 nm used as the reference wavelength for the
    scattering correction. This is the simplest method for correcting for
    scattering, but other methods are available. Users are encouraged to
    explore the other methods and determine which is best for their
    application.

    :param optaa: xarray dataset with the temperature and salinity corrected
        absorbance data array that will be corrected for the effects of
        scattering.
    :param coeffs: Factory calibration coefficients in a dictionary structure

    :return optaa: xarray dataset with the method 1 scatter corrected
        absorbance data array added.
    """
    # find the closest wavelength to 715 nm
    reference_wavelength = 715.0
    idx = np.argmin(np.abs(coeffs['a_wavelengths'] - reference_wavelength))

    # use that wavelength (plue/minus 1 as it isn't exact) as our scatter correction wavelength
    apg_ts = optaa['apg_ts']
    optaa['apg_ts_s'] = apg_ts - apg_ts[:, idx-1:idx+2].max(axis=1)

    return optaa


def estimate_chl_poc(optaa, coeffs, chl_line_height=0.020):
    """
    Derive estimates of Chlorophyll-a and particulate organic carbon (POC)
    concentrations from the temperature, salinity and scatter corrected
    absorption and beam attenuation data.

    :param optaa: xarray dataset with the scatter corrected absorbance data.
    :param coeffs: Factory calibration coefficients in a dictionary structure
    :param chl_line_height: Extinction coefficient for estimating the
        chlorophyll concentration. This value may vary regionally and/or
        seasonally. A default value of 0.020 is used if one is not entered,
        but users may to adjust this based on cross-comparisons with other
        measures of chlorophyll
    :return optaa: xarray dataset with the estimates for chlorophyll and POC
        concentrations added.
    """
    # use the standard chlorophyll line height estimation with an extinction coefficient of 0.020,
    # from Roesler and Barnard, 2013 (doi:10.4319/lom.2013.11.483)
    m650 = np.argmin(np.abs(coeffs['a_wavelengths'] - 650.0))  # find the closest wavelength to 650 nm
    m676 = np.argmin(np.abs(coeffs['a_wavelengths'] - 676.0))  # find the closest wavelength to 676 nm
    m715 = np.argmin(np.abs(coeffs['a_wavelengths'] - 715.0))  # find the closest wavelength to 715 nm
    apg = optaa['apg_ts_s']
    abl = ((apg[:, m715-1:m715+2].median(axis=1) - apg[:, m650-1:m650+2].median(axis=1)) /
           (715 - 650)) * (676 - 650) + apg[:, m650-1:m650+2].median(axis=1)  # interpolate to 676 nm
    aphi = apg[:, m676-1:m676+2].median(axis=1) - abl
    optaa['estimated_chlorophyll'] = aphi / chl_line_height

    # estimate the POC concentration from the attenuation at 660 nm, from Cetinic et al., 2012 and references therein
    # (doi:10.4319/lom.2012.10.415)
    m660 = np.argmin(np.abs(coeffs['c_wavelengths'] - 660.0))  # find the closest wavelength to 660 nm
    cpg = optaa['cpg_ts']
    optaa['estimated_poc'] = cpg[:, m660-1:m660+2].median(axis=1) * 381

    return optaa


def calculate_ratios(optaa):
    """
    Pigment ratios can be calculated to assess the impacts of bio-fouling,
    sensor calibration drift, potential changes in community composition,
    light history or bloom health and age. Calculated ratios are:

    * CDOM Ratio -- ratio of CDOM absorption in the violet portion of the
        spectrum at 412 nm relative to chlorophyll absorption at 440 nm.
        Ratios greater than 1 indicate a preponderance of CDOM absorption
        relative to chlorophyll.
    * Carotenoid Ratio -- ratio of carotenoid absorption in the blue-green
        portion of the spectrum at 490 nm relative to chlorophyll absorption at
        440 nm. A changing carotenoid to chlorophyll ratio may indicate a shift
        in phytoplankton community composition in addition to changes in light
        history or bloom health and age.
    * Phycobilin Ratio -- ratio of phycobilin absorption in the green portion
        of the spectrum at 530 nm relative to chlorophyll absorption at 440 nm.
        Different phytoplankton, notably cyanobacteria, utilize phycobilins as
        accessory light harvesting pigments. An increasing phycobilin to
        chlorophyll ratio may indicate a shift in phytoplankton community
        composition.
    * Q Band Ratio -- the Soret and the Q bands represent the two main
        absorption bands of chlorophyll. The former covers absorption in the
        blue region of the spectrum, while the latter covers absorption in the
        red region. A decrease in the ratio of the intensity of the Soret band
        at 440 nm to that of the Q band at 676 nm may indicate a change in
        phytoplankton community structure. All phytoplankton contain
        chlorophyll 'a' as the primary light harvesting pigment, but green
        algae and dinoflagellates contain chlorophyll 'b' and 'c', respectively,
        which are spectrally redshifted compared to chlorophyll 'a'.

    :param optaa: xarray dataset with the scatter corrected absorbance data.
    :return optaa: xarray dataset with the estimates for chlorophyll and POC
        concentrations added.
    """
    apg = optaa['optical_absorption']
    m412 = np.nanargmin(np.abs(optaa['wavelength_a'].values[0, :] - 412.0))
    m440 = np.nanargmin(np.abs(optaa['wavelength_a'].values[0, :] - 440.0))
    m490 = np.nanargmin(np.abs(optaa['wavelength_a'].values[0, :] - 490.0))
    m530 = np.nanargmin(np.abs(optaa['wavelength_a'].values[0, :] - 530.0))
    m676 = np.nanargmin(np.abs(optaa['wavelength_a'].values[0, :] - 676.0))

    optaa['ratio_cdom'] = apg[:, m412] / apg[:, m440]
    optaa['ratio_carotenoids'] = apg[:, m490] / apg[:, m440]
    optaa['ratio_phycobilins'] = apg[:, m530] / apg[:, m440]
    optaa['ratio_qband'] = apg[:, m676] / apg[:, m440]

    return optaa


class PureWater():
    def __init__(self, purewater_calfile, channel, cal, tscor, ATTRS):
        """
        This object loads, parses, and temperature corrects a pure water
        calibration. A pure water calibration is specific to either the
        a- or c-channels, and should be using the same calibration file
        as the deployed instrument data that you are applying the pure
        water correction to.
        
        :param purewater_calfile: The purewater calibration filepath
        :param channel: Either the a- or c-channel
        :param cal: The calibration coefficients for the deployment
        :param tscor: The temperature-salinity correction matrix
        :param ATTRS: The OPTAA dataset 

        :return self.dat: A dataset object of the loaded and temp-corrected
            purewater calibration
        """
        
        # SAve the channel, temp calibration, and purewater filename
        self.channel = channel
        self.tcal = cal.coeffs["temp_calibration"]
        self.purewater_calfile = purewater_calfile

        # Now load the purewater calibration file  
        self.dat = self.parse_dat_file(self.purewater_calfile, ATTRS)

        # Load & correct the tscor data
        self.tscor = self.load_tscor(tscor)

        # Apply the tscor
        if self.channel == 'a':
            self.dat['a_signal_ts'] = self.apply_tscorr()
        elif self.channel == 'c':
            self.dat['c_signal_ts'] = self.apply_tscorr()

    
    def parse_dat_header(self, header):
        """Parse the header in a dat pure water cal file"""
        # Clean the header
        header = [x.strip() for x in header]
        
        # Timestamp
        timestamp = header[0].split()[-2:]
        timestamp = ' '.join(timestamp)
        
        # Serial Number
        serial_number = [x.split()[0] for x in header if "serial number" in x.lower()][0]
        
        # Output wavelengths
        nwvl = [int(x.split()[0]) for x in header if "output wavelengths" in x.lower()][0]
        
        # Number temperature bins
        ntbins = [int(x.split()[0]) for x in header if "number of temperature bins" in x.lower()][0]
        
        # Temperature bins
        tbins = [float(x) for x in header[-1].split(";")[0].split()]

        return timestamp, serial_number, nwvl, ntbins, tbins


    def parse_dat_file(self, purewater_calfile, ATTRS):

        ##### OPEN THE FILE #####
        with open(purewater_calfile) as file:
            # Read in the header and parse the header data
            header = np.genfromtxt(file, dtype='str', delimiter='\n', max_rows=11)
            timestamp, serial_number, nwvl, ntbins, tbins = self.parse_dat_header(header)
            # Get the wavelengths and associated data
            # Need info from the header file to find the correct rows
            wavelengths = np.genfromtxt(file, dtype='str', delimiter="\n", skip_header=nwvl+2, max_rows=1)
            data = np.genfromtxt(file, dtype="str", delimiter="\n", skip_header=0)


        ##### PARSE THE WAVELENGTHS #####
        wavelengths = [x.strip() for x in str(wavelengths).split()]
        columns = np.concatenate([["elapsed_run_time"], wavelengths, ["Tint", "fw_speed_diagnostic", "pressure", "Text", "abs_ref", "abs_sig", "ccc_ref", "ccc_sig"]])

        ##### PARSE THE DATA INTO A-CHANNEL, C-CHANNEL, AND AUX DATA #####
        data = [x.strip().split() for x in data]

        # Put the data into a dataframe and format values
        df = pd.DataFrame(data, columns=columns)
        df = df.applymap(float)
        # Put the elapsed time into seconds and set as index
        df["elapsed_run_time"] = df["elapsed_run_time"] 
        df.set_index(keys=["elapsed_run_time"], inplace=True)
        # Identify which columns are A-Channel and C-channel
        abs_cols = [v for v in df.columns if "A" in v]
        ccc_cols = [v for v in df.columns if "C" in v]
        aux_cols = [v for v in df.columns if "A" not in v and "C" not in v]
        # Get the A-channel (abs) values
        abs_data = df[abs_cols].values
        # Get the C-channel values
        ccc_data = df[ccc_cols].values
        # Get the auxilary data
        aux_data = df[aux_cols].values

        # Get the wavelength data
        # Generate an interval wavelength number
        int_wavelength = np.arange(0, nwvl, 1)
        # A-channel wavelengths
        abs_wavelength = [float(w.replace("A","0")) for w in abs_cols]
        # C-channel wavelengths
        ccc_wavelength = [float(w.replace("C","0")) for w in ccc_cols]

        # Get the time data
        elapsed_run_time = df.index.values
        dt = pd.to_datetime(timestamp) + pd.to_timedelta(elapsed_run_time, unit='ms')

        # Parse out the aux_data
        Tint = df["Tint"].values
        fw_speed_diagnostic = df["fw_speed_diagnostic"].values
        pressure = df["pressure"].values
        Text = df["Text"].values
        abs_ref = df["abs_ref"].values
        abs_sig = df["abs_sig"].values
        ccc_ref = df["ccc_ref"].values
        ccc_sig = df["ccc_sig"].values

        ##### BUILD A DATASET #####
        ds = xr.Dataset(
            data_vars=dict(
                a_signal=(["time","wavelength_number"], abs_data),
                c_signal=(["time","wavelength_number"], ccc_data),
                wavelength_a=(["wavelength_number"], abs_wavelength),
                wavelength_c=(["wavelength_number"], ccc_wavelength),
                internal_temp=(["time"], Tint),
                external_temp=(["time"], Text),
                filterwheel_speed_diagnostic=(["time"], fw_speed_diagnostic),
                pressure=(["time"], pressure),
                a_reference_dark=(["time"], abs_ref),
                a_signal_dark=(["time"], abs_sig),
                c_reference_dark=(["time"], ccc_ref),
                c_signal_dark=(["time"], ccc_sig),
                elapsed_run_time=(["time"], elapsed_run_time)
            ),
            coords=dict(
                time=dt,
                wavelength_number=int_wavelength
            ),
            attrs=dict(
                comment=("This dataset is for the pure water calibration of either the "
                        "A-channel or C-channel. Please check the filename for which "
                        "side the calibration is applicable for."),
                start_time=timestamp,
                serial_number=serial_number,
                number_temperature_bins=ntbins,
                number_wavelengths=nwvl,
                temperature_bins=tbins
            )
        )

        # Add in variable attributes based on the attributes for the process
        # OPTAA attributes. This is because we need to do a 1:1 mapping of
        # the wavelengths in order to subtract values
        for v in ds.variables:
            if v in ATTRS:
                ds[v].attrs = ATTRS[v]

        return ds
    
    
    def apply_tscorr(self):
        """Function that applies the temperature correction to purewater cals"""

        # Get the external temperature
        nrows = len(self.dat["external_temp"])

        # Get the appropriate temperature correction and associated wavelengths
        Twvl = np.reshape(self.tscor['wvl_cor'], -1)
        if self.channel == 'a':
            wavelengths = self.dat['wavelength_a']
            signal = self.dat['a_signal']
            Tcor = np.reshape(self.tscor["Tcor"], -1)
        elif self.channel == 'c':
            wavelengths = self.dat['wavelength_c']
            signal = self.dat['c_signal']
            Tcor = np.reshape(self.tscor["Tcor"], -1)
        else:
            raise ValueError(f'Channel is {self.channel}. It must be either "a" or "c"')
        
        # Interpolate/extrapolate the correction to the measured wavelengths
        f_Tcor = interp1d(Twvl, Tcor, kind='linear', fill_value='extrapolate')
        temp_cor = f_Tcor(wavelengths)

        # Reshape values to match the shape of the dat channels
        temp_cor = np.tile(temp_cor, [nrows, 1])

        # Calculate the temperature difference
        delta_temp =  np.atleast_2d(self.dat["external_temp"] - self.tcal).T

        # Calculate the temperature correction
        signal_tcor = signal - (delta_temp * temp_cor)

        return signal_tcor
    
    
    def load_tscor(self, tscor):
        """Load the temperature and salinity correction data"""

        TScor = {
            "wvl_cor": [],
            "Tcor": [],
            "a_Scor": [],
            "c_Scor": []
        }

        for key in tscor:
            # Get the values
            Tcor, c_Scor, a_Scor = tscor.get(key)
            if np.isnan(Tcor):
                continue
            else:
                TScor["wvl_cor"].append(key)
                TScor["Tcor"].append(Tcor)
                TScor["c_Scor"].append(c_Scor)
                TScor["a_Scor"].append(a_Scor)

        return TScor