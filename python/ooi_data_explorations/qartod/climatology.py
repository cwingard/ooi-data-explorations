import numpy as np
import pandas as pd
import xarray as xr


class Climatology():

    def std(self, da):
        """Calculate the standard deviation of grouped-monthly data.

        Calculates the standard deviation for a calendar-month from all
        of the observations for a given calendar-month. Does linear
        interpolation to fill NaNs. Performs a 90-degree phase shift to interp
        edge-cases.

        Parameters
        ----------
        ds: (xarray.DataSet)
            DataSet of the original time series observations
        param: (str)
            A string corresponding to the variable in the DataSet which is fit.

        Attributes
        ----------
        monthly_std: (pandas.Series)
            The standard deviation for a calendar month calculated from all of
            the observations for a given calendar-month.
        """
        da = da.groupby(da.time.dt.month).std()
        keys = list(da.dims)
        if 'depth' in keys:
            # calculate the mean standard deviation for all the depths
            self.monthly_std = pd.Series(da.mean(dim='depth'), index=da.month)
        else:
            self.monthly_std = pd.Series(da.values, index=da.month)

        # Fill missing std values
        ind = np.arange(1, 13, 1)
        self.monthly_std = self.monthly_std.reindex(index=ind)
        # First, interpolate only NaNs surrounded by valid values
        self.monthly_std = self.monthly_std.interpolate(limit_area="inside")
        # Second, shift values by 3 (90 degrees) to capture edge cases
        self.monthly_std = self.monthly_std.reindex(index=np.roll(
            self.monthly_std.index, 3)).interpolate(limit_area="inside")
        # Finally, shift values back by 3 (90 degrees) to reset index
        self.monthly_std = self.monthly_std.reindex(index=np.roll(self.monthly_std.index, -3))

    def mu(self, da):
        """
        Calculates the mean for a calendar-month from all observations for a
        given calendar-month. Does linear interpolation to fill NaNs. Performs
        a 90-degree phase shift to interp edge-cases.

        Parameters
        ----------
        ds: (xarray.DataSet)
            DataSet of the original time series observations
        param: (str)
            A string corresponding to the variable in the DataSet which is fit.

        Attributes
        ----------
        monthly_mu: (pandas.Series)
            The mean for a calendar month calculated from all observations for
            a given calendar-month.
        """
        da = da.groupby(da.time.dt.month).mean()
        keys = list(da.dims)
        if 'depth' in keys:
            # calculate the mean for all the depths
            self.monthly_mu = pd.Series(da.mean(dim='depth'), index=da.month)
        else:
            self.monthly_mu = pd.Series(da.values, index=da.month)

        # Fill missing std values
        ind = np.arange(1, 13, 1)
        self.monthly_mu = self.monthly_mu.reindex(index=ind)
        # First, interpolate only NaNs surrounded by valid values
        self.monthly_mu = self.monthly_mu.interpolate(limit_area="inside")
        # Second, shift values by 3 (90 degrees) to capture edge cases
        self.monthly_mu = self.monthly_mu.reindex(index=np.roll(
            self.monthly_mu.index, 3)).interpolate(limit_area="inside")
        # Finally, shift values back by 3 (90 degrees) to reset index
        self.monthly_mu = self.monthly_mu.reindex(index=np.roll(self.monthly_mu.index, -3))

    def fit(self, da):
        """Calculate the climatological fit and monthly standard deviations.

        Calculates the climatological fit for a time series. First, the data
        are binned by month and averaged. Next, a four-cycle harmonic is fitted
        via OLS-regression. The climatological expected value for each month
        is then calculated from the regression coefficients using only the first
        two cycles from the harmonic fit. Finally, the standard deviation is
        derived using all the observations for a given month.

        Parameters
        ----------
        ds: (xarray.DataArray)
            DataArray of the original time series observations
       
        Attributes
        -------
        fitted_data: (pandas.Series)
            The climatological monthly expectation calculated from the
            regression, indexed by the year-month
        regression: (dict)
            A dictionary containing the OLS-regression values for
            * beta: Least-squares solution.
            * residuals: Sums of residuals; squared Euclidean 2-norm
            * rank: rank of the input matrix
            * singular_values: The singular values of input matrix
            * variance_explained: coefficient of determination, or R^2
        monthly_fit: (pandas.Series)
            The climatological expectation for each calendar month of a year

        Example
        -------
        from qartod.climatology import Climatology
        climatology = Climatology()
        climatology.fit(ctdbp_data)
        """
        # Resample the data to monthly means
        mu = da.resample(time="M").mean()
        keys = list(mu.dims)
        if 'depth' in keys:
            mu = mu.mean(dim='depth')

        # Next, build the model
        ts = mu.values
        f = 1/12
        N = len(ts)
        t_in = np.arange(0, N, 1)
        t_out = t_in

        # Drop NaNs from the fit
        mask = np.isnan(ts)
        ts = ts[mask == False]
        t_in = t_in[mask == False]
        n = len(t_in)

        # build the four-cycle model
        X = [np.ones(n), np.sin(2 * np.pi * f * t_in), np.cos(2 * np.pi * f * t_in),
             np.sin(4 * np.pi * f * t_in), np.cos(4 * np.pi * f * t_in),
             np.sin(6 * np.pi * f * t_in), np.cos(6 * np.pi * f * t_in),
             np.sin(8 * np.pi * f * t_in), np.cos(8 * np.pi * f * t_in)]

        [beta, resid, rank, s] = np.linalg.lstsq(np.transpose(X), ts, rcond=-1)
        self.regression = {
            "beta": beta,
            "residuals": resid,
            "rank": rank,
            "singular_values": s,
            "variance_explained": 1 - resid / sum((ts - ts.mean()) ** 2)
        }

        if self.regression['variance_explained'] > 0.15:
            # Calculate the two-cycle fitted data
            fitted_data = beta[0] + beta[1]*np.sin(2*np.pi*f*t_out) + beta[2]*np.cos(
                2*np.pi*f*t_out) + beta[3]*np.sin(4*np.pi*f*t_out) + beta[4]*np.cos(4*np.pi*f*t_out)

            fitted_data = pd.Series(fitted_data, index=mu.get_index("time"))
            self.fitted_data = fitted_data

            # Return the monthly_avg from the fitted data
            self.monthly_fit = self.fitted_data.groupby(self.fitted_data.index.month).mean()
        else:
            # Return the monthly_avg from the monthly means
            self.mu(da)
            self.monthly_fit = self.monthly_mu

        # Set self.monthly_std
        self.std(da)
