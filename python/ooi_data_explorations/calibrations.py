#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Provides common methods for working with calibration coefficients from the OOI M2M API
"""
import json
import numpy as np

from ooi_data_explorations.common import get_calibrations_by_uid


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types, where we have nested numpy arrays in
    a dictionary. Allows saving the data to a json file. Used by the
    Coefficients and Blanks class to save instrument calibration coefficients
    to disk

    From our trusty friends at StackOverflow: https://stackoverflow.com/a/49677241
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Coefficients(object):
    """
    A Coefficients class with two methods to load/save the serialized calibration coefficients for an instrument.
    """
    def __init__(self, coeff_file):
        """
        Initialize the class with the path to coefficients file and an empty dictionary structure for
        the calibration data
        """
        # set the infile name and path
        self.coeff_file = coeff_file
        self.coeffs = {}

    def load_coeffs(self):
        """
        Obtain the calibration data for this instrument from a JSON data file.
        """
        with open(self.coeff_file, 'r') as f:
            coeffs = json.load(f)

        # JSON loads arrays as lists. We need to convert those to arrays for our work
        for c in coeffs:
            if isinstance(coeffs[c], list):
                coeffs[c] = np.asarray(coeffs[c])

        self.coeffs = coeffs

    def save_coeffs(self):
        """
        Save the calibration data for this instrument to a JSON data file.
        """
        with open(self.coeff_file, 'w') as f:
            jdata = json.dumps(self.coeffs, cls=NumpyEncoder)
            f.write(jdata)


def compare_names(cal_name, data_source):
    """
    Internal function to compare the calibration data file name to the data
    source name. If the names are inconsistent, raise an error.

    :param cal_name: name of the calibration data file
    :param data_source: name of the data source
    :return: name of the calibration data file
    """
    if not cal_name:
        cal_name = data_source
    else:
        if cal_name != data_source:
            raise ValueError('Calibration data file name inconsistent, unable to properly parse the '
                             'calibration data.')

    return cal_name


def load_cal_coefficients(cal_file, calibrations, uid, start_time):
    """
    Load the calibration coefficients for the instrument and deployment from
    the OOI M2M system or from a local file. If the local file does not exist,
    the calibration coefficients will be downloaded from the OOI M2M system and
    saved to the local file for future use.

    :param cal_file: path to the local calibration file
    :param calibrations: calibration class specific to the instrument type
    :param uid: unique identifier (UID) of the instrument
    :param start_time: deployment start time in seconds since 1970-01-01
    :return: calibration coefficients dictionary
    """
    # load the instrument calibration data
    dev = calibrations(cal_file)  # initialize calibration class

    # check for the source of calibration coeffs and load accordingly
    if os.path.isfile(cal_file):
        # we always want to use this file if it already exists
        dev.load_coeffs()
    else:
        # load from the OOI M2M system and create a list of calibration events relative to the deployment start date
        cals = get_calibrations_by_uid(uid)
        cal_idx = {}
        for cal in cals['calibration']:
            # for each instance of a calibration event for this instrument...
            tdiff = []
            for data in cal['calData']:
                # calculate the time difference between the start of the deployment and the calibration event
                td = (data['eventStartTime'] / 1000) - start_time
                if td <= 0:
                    # valid cals must come before the deployment start date/time
                    tdiff.append(td)
                else:
                    # use a ridiculously large number to avoid selecting this cal event
                    tdiff.append(10**20)

            # find the calibration event closest to the start of the deployment
            cal_idx[cal['name']] = np.argmin(np.abs(tdiff))

        # load the calibration coefficients
        dev.parse_m2m_cals(cals['serialNumber'], cals['calibration'], cal_idx)

        # save the calibration coefficients
        dev.save_coeffs()

    return dev
