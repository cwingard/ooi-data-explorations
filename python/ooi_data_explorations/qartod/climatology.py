#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Andrew Reed
@brief Used to calculate the monthly climatology values for QARTOD
"""

import numpy as np
import pandas as pd
import xarray as xr


class Climatology():
    """
    Calculate monthly climatology statistics and harmonic fits from time series data.

    This class computes monthly means, standard deviations, and fits a harmonic
    model to seasonal data, with special handling for NaNs and edge cases.
    """

    def _interpolate_monthly_series(self, series: pd.Series) -> pd.Series:
        """
        Interpolate a monthly Series with handling for edge NaNs via phase shift.

        Parameters
        ----------
        series : pandas.Series
            A 12-element series indexed by calendar month (1–12).

        Returns
        -------
        pandas.Series
            Interpolated monthly series, same index (1–12).
        """
        full_index = np.arange(1, 13)
        series = series.reindex(index=full_index)

        # Inside-only interpolation
        series = series.interpolate(limit_area="inside")

        # Edge-case interpolation using 90-degree phase shift (3 months)
        series = series.reindex(index=np.roll(series.index, 3))
        series = series.interpolate(limit_area="inside")
        series = series.reindex(index=np.roll(series.index, -3))

        return series
        

    def _compute_monthly_stat(self, da: xr.DataArray, stat: str) -> pd.Series:
        """
        Compute a monthly statistic (mean or std) from a DataArray.

        Parameters
        ----------
        da : xarray.DataArray
            Time series data containing a 'time' coordinate.
        stat : {"mean", "std"}
            The statistic to compute.

        Returns
        -------
        pandas.Series
            Monthly statistic indexed by calendar month (1–12).
        """
        grouped = getattr(da.groupby(da.time.dt.month), stat)()
        if 'depth' in grouped.dims:
            values = grouped.mean(dim='depth').values
        else:
            values = grouped.values
        return pd.Series(values, index=grouped.month.values)
        

    def std(self, da: xr.DataArray):
        """
        Compute and interpolate monthly standard deviation.

        Parameters
        ----------
        da : xarray.DataArray
            Time series data with 'time' coordinate and optional 'depth' dimension.

        Attributes
        ----------
        monthly_std : pandas.Series
            Interpolated monthly standard deviation for months 1–12.
        """
        std_series = self._compute_monthly_stat(da, 'std')
        self.monthly_std = self._interpolate_monthly_series(std_series)
        

    def mu(self, da: xr.DataArray):
        """
        Compute and interpolate monthly mean.

        Parameters
        ----------
        da : xarray.DataArray
            Time series data with 'time' coordinate and optional 'depth' dimension.

        Attributes
        ----------
        monthly_mu : pandas.Series
            Interpolated monthly mean for months 1–12.
        """
        mu_series = self._compute_monthly_stat(da, 'mean')
        self.monthly_mu = self._interpolate_monthly_series(mu_series)


    def fit(self, da: xr.DataArray):
        """
        Calculate climatological monthly fit using 2-cycle harmonic regression.

        The method first resamples the data to monthly means, then fits a
        harmonic model with up to 4 annual cycles using OLS regression.
        If the model explains more than 15% of variance (R² > 0.15), the
        fitted two-cycle model is used for the monthly climatology.
        Otherwise, the raw monthly means are used. Standard deviations are
        always computed from the original data.

        Parameters
        ----------
        da : xarray.DataArray
            Time series data with 'time' coordinate and optional 'depth' dimension.

        Attributes
        ----------
        monthly_fit : pandas.Series
            Monthly climatological expectation from harmonic fit or mean.
        monthly_std : pandas.Series
            Monthly standard deviation.
        fitted_data : pandas.Series
            Time-indexed fitted values from harmonic regression (if applied).
        regression : dict
            OLS regression details:
                - beta : ndarray
                    Regression coefficients.
                - residuals : ndarray
                    Sum of squared residuals.
                - rank : int
                    Rank of the design matrix.
                - singular_values : ndarray
                    Singular values of the design matrix.
                - variance_explained : float
                    Coefficient of determination (R²).

        Notes
        -----
        The harmonic fit uses:
            y(t) = β₀ + β₁ sin(2πft) + β₂ cos(2πft)
                  + β₃ sin(4πft) + β₄ cos(4πft)
        where f = 1/12 cycles per month.
        """
        mu = da.resample(time="M").mean()
        if 'depth' in mu.dims:
            mu = mu.mean(dim='depth')

        ts = mu.values
        time_index = mu.get_index("time")
        f = 1.0 / 12
        t = np.arange(len(ts))

        # Drop NaNs
        valid = ~np.isnan(ts)
        ts_valid = ts[valid]
        t_valid = t[valid]

        # If too few valid points, fallback to mean
        if len(ts_valid) < 5:
            self.mu(da)
            self.monthly_fit = self.monthly_mu
            self.std(da)
            return

        # Build harmonic regression matrix (4 cycles)
        X = np.column_stack([
            np.ones_like(t_valid),
            np.sin(2 * np.pi * f * t_valid), np.cos(2 * np.pi * f * t_valid),
            np.sin(4 * np.pi * f * t_valid), np.cos(4 * np.pi * f * t_valid),
            np.sin(6 * np.pi * f * t_valid), np.cos(6 * np.pi * f * t_valid),
            np.sin(8 * np.pi * f * t_valid), np.cos(8 * np.pi * f * t_valid),
        ])

        beta, resid, rank, s = np.linalg.lstsq(X, ts_valid, rcond=None)
        total_ss = np.sum((ts_valid - ts_valid.mean()) ** 2)
        resid_sum = resid if resid.size else 0.0
        r_squared = 1.0 - (resid_sum / total_ss if total_ss > 0 else 0.0)
        self.regression = {
            "beta": beta,
            "residuals": resid,
            "rank": rank,
            "singular_values": s,
            "variance_explained": r_squared,
        }

        # calculate the monthly averages first from the monthly means
        self.mu(da)
        self.monthly_fit = self.monthly_mu
        if self.regression['residuals'].size > 0:
            if self.regression['variance_explained'] > 0.15:
                # Calculate the two-cycle fitted data
                t_out = np.arange(len(ts))
                fitted_data = beta[0] + beta[1]*np.sin(2*np.pi*f*t_out) + beta[2]*np.cos(
                    2*np.pi*f*t_out) + beta[3]*np.sin(4*np.pi*f*t_out) + beta[4]*np.cos(4*np.pi*f*t_out)

                fitted_data = pd.Series(fitted_data, index=mu.get_index("time"))
                self.fitted_data = fitted_data

                # re-calculate the monthly averages using the fitted data
                self.monthly_fit = self.fitted_data.groupby(self.fitted_data.index.month).mean()

        self.std(da)
