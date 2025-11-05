#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Andrew Reed
@brief Used to calculate the gross range user and sensor ranges for QARTOD
"""

import numpy as np
import xarray as xr
from typing import Tuple
import dask
from dask.diagnostics import ProgressBar
from scipy.stats import normaltest, shapiro


class GrossRange:
    """
    Gross Range fitting process for QARTOD.

    For a given parameter in a dataset, calculate the gross range QARTOD values
    for the data, and format the data into a qcConfig object optimized for use
    with Axiom-implemented QARTOD gross range test.

    Examples
    --------
    >>> gross_range = GrossRange(fail_min=200, fail_max=1200)
    >>> gross_range.fit(pco2_dataset, "pco2_seawater")
    >>> gross_range.make_qcConfig()
    >>> gross_range.qcConfig
    {'qartod': {'gross_range_test': {'suspect_span': [200, 767.5], 'fail_span': [200, 1200]}}}
    """

    def __init__(self, fail_min: float, fail_max: float):
        """
        Initialize the Gross Range with fail min/max thresholds.

        Parameters
        ----------
        fail_min : float
            Minimum allowed value; anything below is considered a fail.
        fail_max : float
            Maximum allowed value; anything above is considered a fail.
        """
        if fail_min >= fail_max:
            raise ValueError("fail_min must be less than fail_max")
        self.fail_min = fail_min
        self.fail_max = fail_max


    def fit(self, ds: xr.Dataset, param: str, sigma: float = 3,
            check_normality: bool = False, **kwargs):
        """
        Fit suspect range based on standard deviations or percentiles.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing the variable to fit.
        param : str
            Name of the variable in `ds` to evaluate.
        sigma : float, default 3
            Number of standard deviations for calculating the suspect range.
        check_normality : bool, default False
            If True, test for normality and use percentiles if non-normal.

        Side effects
        ------------
        Stores attributes on the instance:
        - suspect_min, suspect_max: float (rounded to 5 decimals)
        - source: str (description)
        """
        # Check that the param is in the dataset
        if param not in ds:
            raise KeyError(f"Parameter '{param}' not found in dataset")

        # Filter out the fail-range values and flatten to 1-D array
        da = self.filter_fail_range(ds[param])
        arr = self._to_1d_numpy(da)

        # Decide whether to use std-dev or percentile method
        use_normal_stats = True
        if check_normality:
            print(f"----- Testing {param} data for normality -----")
            pnorm = self.check_normality(arr)
            if np.isnan(pnorm):
                # Normality test didn't run
                use_normal_stats = False
                print("----- Normality test returned NaN; using percentile fallback -----")
            elif pnorm < 0.01:
                use_normal_stats = False
                print(f"----- {param} is not normally distributed (p={pnorm:.3e}) -----")
            else:
                print(f"----- {param} appears approximately normal (mean p={pnorm:.3e}) -----")

        if use_normal_stats:
            suspect_min, suspect_max = self._suspect_from_std(arr, sigma)
            source = (f"User range based on mean ± {sigma} standard deviations "
                      "of all observations.")
        else:
            suspect_min, suspect_max, p_range = self._suspect_from_percentiles(arr, sigma)
            source = (f"User range based on percentiles covering {p_range}% of "
                      "data, approximating the Empirical Rule.")

        # Ensure suspect range is within fail range
        suspect_min = max(suspect_min, self.fail_min)
        suspect_max = min(suspect_max, self.fail_max)

        # Store results
        self.suspect_min = np.round(suspect_min, 5)
        self.suspect_max = np.round(suspect_max, 5)
        self.source = source


    def filter_fail_range(self, da: xr.DataArray) -> xr.DataArray:
        """
        Filter out values outside the fail range.

        Parameters
        ----------
        da : xarray.DataArray
            Input data.

        Returns
        -------
        xarray.DataArray
            Filtered data array.
        """
        return da.where((da > self.fail_min) & (da < self.fail_max), drop=True)


    def _suspect_from_std(self, arr: np.ndarray, sigma: float) -> Tuple[float, float]:
        """
        Calculate suspect min/max using mean ± sigma * std.

        Parameters
        ----------
        arr : np.ndarray
            1D array of values (NaNs already removed).
        sigma : float
            Number of standard deviations.

        Returns
        -------
        (suspect_min, suspect_max)
        """
        avg = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        return avg - sigma * std, avg + sigma * std


    def _suspect_from_percentiles(self, arr: np.ndarray, sigma: float) -> Tuple[float, float, float]:
        """
        Calculate suspect min/max from percentiles approximating the Empirical Rule.

        Parameters
        ----------
        arr : np.ndarray
            1D array of values (NaNs already removed).
        sigma : float
            Number of standard deviations to approximate (3, 4, or 5 supported).

        Returns
        -------
        (suspect_min, suspect_max, p_range)
            suspect_min, suspect_max : floats from percentiles
            p_range : float - percent of data covered by the chosen percentiles

        Notes
        -----
        Percentile values are treated as percentages (0-100). For example,
        0.15 corresponds to the 0.15th percentile (tail for 3σ).
        """
        percentile_map = {
            3: 0.15,        # 0.15th percentile -> covers 99.7% (≈ 3σ)
            4: 0.003167,    # 0.003167th percentile -> ≈ 4σ two-tailed
            5: 0.00002867   # 0.00002867th percentile -> ≈ 5σ two-tailed
        }
        lower_p = percentile_map.get(sigma)
        if lower_p is None:
            raise ValueError(f"No percentile mapping for sigma={sigma}; supported: {list(percentile_map)}")

        upper_p = 100.0 - lower_p
        
        # Use numpy's nanpercentile (expects percent in 0-100 range)
        suspect_min = float(np.nanpercentile(arr, lower_p))
        suspect_max = float(np.nanpercentile(arr, upper_p))
        p_range = float(np.round(100.0 - lower_p * 2.0, 1))
        print(f"Using percentiles from {lower_p} to {upper_p}")
        
        return suspect_min, suspect_max, p_range


    def check_normality(self, arr: numpy.ndarray, n_choices: int = 5000, n_iters: int = 1000) -> float:
        """
        Check if the data is normally distributed using repeated sampling.

        If the len(arr) <= 5000: use scipy.stats.shapiro to test normality
        If the len(arr) > 5000:
            * Randomly choose 

        Parameters
        ----------
        arr : np.ndarray
            1D array of values (NaNs already removed).
        n_choices : int, default 5000
            Number of random samples in each iteration.
        n_iters : int, default 1000
            Number of iterations.

        Returns
        -------
        float
            Mean p-value from normality tests across iterations.
        """
        # Check the size of the dataset
        n = arr.size
        if n == 0:
            return float('nan')
            
        # If dataset is small enough, don't need to subsample
        if n <= 5000:
            try:
                stat, pval = shapiro(arr)
                return float(pval)
            except Exception:
                # Fallback to normaltest
                try:
                    stat, pval = normaltest(arr)
                    return float(arr)
                except Exception:
                    return float('nan')
        # Otherwise, need to random subsample and parallelize to speed up calc
        else:
            vals = [
                dask.delayed(np.random.choice)(arr, n_choices)
                for _ in range(n_iters)
            ]
            with ProgressBar():
                samples = dask.compute(*vals)
            pvals = [normaltest(v).pvalue for v in samples]
            return np.mean(pvals)


    def make_qcConfig(self):
        """
        Build a formatted qcConfig dictionary for QARTOD gross range test.

        Attributes
        ----------
        qcConfig : dict
            Nested dict formatted for QARTOD gross range test. Requires that
            `fit()` has already been executed.
        """
        if not hasattr(self, "suspect_min") or not hasattr(self, "suspect_max"):
            raise RuntimeError("Must run fit(...) before make_qcConfig()")
            
        self.qcConfig = {
            "qartod": {
                "gross_range_test": {
                    "suspect_span": [self.suspect_min, self.suspect_max],
                    "fail_span": [self.fail_min, self.fail_max]
                }
            }
        }

    # -----------------------
    # Utility helpers
    # -----------------------
    @staticmethod
    def _to_1d_numpy(x) -> np.ndarray:
        """
        Convert xarray.DataArray or numpy-like input to a 1D numpy array with NaNs removed.
        """
        if isinstance(x, xr.DataArray):
            arr = x.values
        else:
            arr = np.asarray(x)
        arr = arr.ravel()
        arr = arr[~np.isnan(arr)]
        return arr

