import os
import numpy as np
import pandas as pd
import xarray as xr


class GrossRange():
    """Gross Range fitting process for QARTOD.

    For a given parameter in a dataset, calculate the gross range QARTOD values
    for the data, and fromat the data into a qcConfig object optimized for use
    with Axiom-implemented QARTOD gross range test.

    Example
    -------
    from qartod.gross_range import GrossRange
    gross_range = GrossRange(fail_min=200, fail_max=1200)
    gross_range.fit(pco2_dataset, "pco2_seawater")
    gross_range.make_qcConfig()
    gross_range.qcConfig = {'qartod':
        {'gross_range_test':
            {'suspect_span': [200, 767.5], 'fail_span': [200, 1200]}}}
    """

    def __init__(self, fail_min, fail_max):
        """Init the Gross Range with the relevant fail min/max.

        Parameters
        ----------
        fail_min: (float)
            The minimum value for the given dataset parameter below which the
            data is fail.
        fail_max: (float)
            The maximum value for the given dataset parameter above which the
            data is fail.
        """
        self.fail_min = fail_min
        self.fail_max = fail_max

    def fit(self, ds, param, sigma=5):
        """Fit suspect range with specified standard deviation.

        Parameters
        ----------
        ds: (xarray.DataSet)
            An xarray datasets containing the given data variable to be fitted,
            with a primary dimension of time.
        param: (string)
            The name of the data variable from the given dataset to fit
        sigma: (float)
            The number of standard deviations for calculating the suspect range

        """
        # First, filter out data which falls outside of the fail ranges
        ds = self.filter_fail_range(ds, param)

        # Calculate the mean and standard deviation
        avg = np.nanmean(ds[param])
        std = np.nanstd(ds[param])

        # Calculate the suspect range
        suspect_min = avg-sigma*std
        suspect_max = avg+sigma*std

        # If the suspect ranges are outside the fail ranges, set
        # suspect ranges to the fail_ranges
        if suspect_min < self.fail_min:
            suspect_min = self.fail_min
        if suspect_max > self.fail_max:
            suspect_max = self.fail_max

        # Save the results
        self.suspect_min = np.round(suspect_min, decimals=2)
        self.suspect_max = np.round(suspect_max, decimals=2)

    def filter_fail_range(self, ds, param):
        """Filter out values which fall outside the fail range."""
        ds = ds.where((ds[param] < self.fail_max) &
                      (ds[param] > self.fail_min), drop=True)
        return ds

    def make_qcConfig(self):
        """Build properly formatted qcConfig object for qartod gross range."""
        self.qcConfig = {
            "qartod": {
                "gross_range_test": {
                    "suspect_span": [self.suspect_min, self.suspect_max],
                    "fail_span": [self.fail_min, self.fail_max]
                }
            }
        }
