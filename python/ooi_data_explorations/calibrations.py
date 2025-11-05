#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Provides common methods for working with calibration coefficients from the OOI M2M API
"""
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types, where we have nested numpy arrays in
    a dictionary. Allows saving the data to a json file. Used by the
    Coefficients and Blanks class to save instrument calibration coefficients
    to disk

    From our trusty friends at StackOverflow: https://stackoverflow.com/a/49677241
    """
    def default(self, obj):
        if isinstance(obj, (np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
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
