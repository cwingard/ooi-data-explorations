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


class Calibrations(Coefficients):
    def __init__(self, coeff_file):
        """
        Loads the NUTNR calibration coefficients for a unit. Values come from
        either a serialized object created per instrument and deployment
        (calibration coefficients do not change in the middle of a deployment),
        or from the calibration data available from the OOI M2M API.
        """
        # assign the inputs
        Coefficients.__init__(self, coeff_file)
        self.coeffs = None

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
            # calibration temperature
            if cal['name'] == 'CC_cal_temp':
                coeffs['cal_temp'] = cal['calData'][cal_idx['CC_cal_temp']]['value']
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_cal_temp']]['dataSource'])

            # absorption offsets for pure water
            if cal['name'] == 'CC_di':
                coeffs['pure_water'] = np.array(cal['calData'][cal_idx['CC_di']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_di']]['dataSource'])

            # NO3 absorption coefficients
            if cal['name'] == 'CC_eno3':
                coeffs['a_wavelengths'] = np.array(cal['calData'][cal_idx['CC_awlngth']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_awlngth']]['dataSource'])
            # Seawater absorption coefficients
            if cal['name'] == 'CC_eswa':
                coeffs['c_wavelengths'] = np.array(cal['calData'][cal_idx['CC_cwlngth']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_cwlngth']]['dataSource'])

            # upper and lower wavelength limits for the spectra fit
            if cal['name'] == 'CC_lower_wavelength_limit_for_spectra_fit':
                coeffs['limits'] = [cal['calData'][cal_idx['CC_lower_wavelength_limit_for_spectra_fit']]['value'],
                                    cal['calData'][cal_idx['CC_upper_wavelength_limit_for_spectra_fit']]['value']]
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_lower_wavelength_limit_for_spectra_fit']]
                                         ['dataSource'])

            # wavelengths measured by the instrument
            if cal['name'] == 'CC_wl':
                coeffs['wlngths'] = np.array(cal['calData'][cal_idx['CC_wl']]['value'])
                cal_name = compare_names(cal_name, cal['calData'][cal_idx['CC_wl']]['dataSource'])

        # serial number, stripping off all but the numbers
        coeffs['serial_number'] = int(re.sub('[^0-9]', '', serial_number))

        # save the resulting dictionary
        self.coeffs = coeffs

