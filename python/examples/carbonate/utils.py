#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Common functions used in the evaluation of the carbonate data sets
    in preparation for developing analysis-ready data products
"""
import numpy as np
import pandas as pd
import warnings

from datetime import datetime, UTC
from erddapy import ERDDAP
from gsw import p_from_z, SA_from_SP, pot_rho_t_exact
from scipy import odr
from sklearn.linear_model import LinearRegression, RANSACRegressor

from ooi_data_explorations.common import list_deployments, get_sensor_information, add_annotation_qc_flags, dict_update
from ooi_data_explorations.qartod.discrete_samples import get_discrete_samples
from ooi_data_explorations.bottles import clean_data

from global_metadata import SHARED


def apply_qc_results(ds, annotations):
    """
    Apply quality control results to dataset by NaN-ing failed data points.

    Uses QARTOD test results, instrument-specific QC flags, and human-in-the-loop
    (HITL) annotations to mask data points that failed quality control checks.
    Data points with a QC flag value of 4 (fail) are converted to NaN.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables with associated QC test results and flags.
    annotations : list of dict
        List of annotation dictionaries from the OOI API containing manual QC
        flags and operational status information.

    Returns
    -------
    xarray.Dataset
        The dataset with failed data points converted to NaN based on QC results
        and annotations. Records where the rollup annotation is set to fail are
        completely removed from the dataset.

    Notes
    -----
    The function processes QC results in the following order:
    1. QARTOD automated test results (variables ending in '_qartod_results')
    2. Instrument-specific QC flags (variables ending in '_quality_flag')
    3. Variable-specific HITL annotations (variables ending in '_annotations_qc_results')
    4. Rollup annotations ('rollup_annotations_qc_results')

    Annotations marked as 'not_operational', 'not_available', or 'pending_ingest'
    are excluded. The function makes up to 3 attempts to add annotations to handle
    potential gateway errors.
    """
    # create a list of any variables that have had qartod tests applied
    variables = [x.split('_qartod_results')[0] for x in ds.variables if 'qartod_results' in x]

    # if we have any tests results, NaN out the ones that failed
    if variables:
        for v in variables:
            ds[v] = ds[v].where(ds[v + '_qartod_results'] != 4)

    # create a list of any variables that have had instrument specific tests
    variables = [x.split('_quality_flag')[0] for x in ds.variables if '_quality_flag' in x]

    # if we have any tests results, NaN out the ones that failed
    if variables:
        for v in variables:
            ds[v] = ds[v].where(ds[v + '_quality_flag'] != 4)

    # convert the annotations to a dataframe and remove the operational notes and any periods where data was unavailable
    annotations = pd.DataFrame(annotations)
    if not annotations.empty:
        annotations = annotations[~annotations['qcFlag'].isin([None, 'not_operational', 'not_available', 'pending_ingest'])]

    # now add the annotations to the data set (if any still available)
    if not annotations.empty:
        for n in range(3):  # try to add the annotations (handle gateway errors)
            try:
                added = add_annotation_qc_flags(ds, annotations)
                ds = added.copy()
                break
            except Exception as e:
                if n < 2:  # try 3 times to get annotations added
                    print(f'Trying to add annotations failed, attempt {n+1} of 3')
                    continue
                else:
                    warnings.warn(f'Unable to add annotations to the dataset due to {e}')

    # create a list of any variables that have had individual annotation flags assigned
    variables = [x.split('_annotations_qc_results')[0] for x in ds.variables if '_annotations_qc_results' in x]

    # if we have any variable specific HITL annotations, NaN out the ones that were set to fail or missing
    if variables:
        for v in variables:
            if v in ds.variables:
                ds[v] = ds[v].where(ds[v + '_annotations_qc_results'] != 4)

    # finally, remove data where the rollup annotation is set to fail
    if 'rollup_annotations_qc_results' in ds.variables:
        ds = ds.where(ds['rollup_annotations_qc_results'] != 4, drop=True)

    # return the cleaned-up record with annotations (if any, added)
    return ds


def calculate_alkalinity(ds, salinity, temperature, model='AF'):
    """
    Calculate estimated total alkalinity from salinity and temperature.

    Estimates total alkalinity using one of three empirical models based on
    salinity alone or salinity and temperature measurements.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to update with calculated alkalinity values.
    salinity : str
        Name of the salinity variable in the dataset.
    temperature : str
        Name of the temperature variable in the dataset.
    model : {'AF', 'KL', 'TT'}, optional
        Alkalinity estimation model to use (default is 'AF'):

        - 'AF': Fassbender et al. (2017) - salinity-based linear model
        - 'KL': Lee et al. (2006) - temperature and salinity polynomial model
        - 'TT': Takahashi et al. (2018) - salinity-based linear model

    Returns
    -------
    xarray.Dataset
        The dataset with a new 'alkalinity' variable added containing the
        estimated total alkalinity values.

    Notes
    -----
    If an invalid model is specified, the alkalinity variable is created but
    filled with NaN values. The 'KL' model requires the dataset to have a
    'longitude' variable for the calculation.

    References
    ----------
    Fassbender, A. J., et al. (2017). Seasonal carbonate chemistry variability
    in marine surface waters of the US Pacific Northwest. Earth System Science
    Data, 9(1), 45-60.

    Lee, K., et al. (2006). Global relationships of total alkalinity with
    salinity and temperature in surface waters of the world's oceans.
    Geophysical Research Letters, 33(19).

    Takahashi, T., et al. (2018). Climatological distributions of pH, pCO2,
    total CO2, alkalinity, and CaCO3 saturation in the global surface ocean.
    Marine Chemistry.
    """
    if model not in ['KL', 'AF', 'TT']:
        ds['alkalinity'] = ds[salinity] * np.nan

    if model == 'KL':
        # calculate an estimated alkalinity product from the temperature and salinity records
        ds['alkalinity'] = (2305 + 53.23 * (ds[salinity] - 35) + 1.85 * (ds[salinity] - 35)**2 -
                            14.72 * (ds[temperature] - 20) - 0.158 * (ds[temperature] - 20)**2 +
                            0.062 * (ds[temperature] - 20) * ds['longitude'])
    
    if model == 'AF':
        ds['alkalinity'] = 47.7 * ds[salinity] + 647.0
    
    if model == 'TT':
        ds['alkalinity'] = 44.88 * ds[salinity] + 724.8

    return ds
    

def load_discrete(depth, depth_limit=3.0):
    """
    Load the discrete sample data for the Endurance Array (including Cabled
    Array data) to use in validating and potentially calibrating the in-situ
    data.
    """
    # download the discrete sample data collected by the Endurance and Cabled Array (their sampling overlaps with two of our sites)
    discrete = [get_discrete_samples('Endurance'), get_discrete_samples('Cabled')]
    discrete = pd.concat(discrete, ignore_index=True)
    discrete = clean_data(discrete)  # convert WOCE-style flags to their QARTOD-style equivalent
    
    # limit the discrete sample data in depth
    discrete = discrete[(discrete['CTD Depth [m]'] - depth).abs() < depth_limit]

    # fill in missing CTD pressure data values (small boat samples)
    dbar = p_from_z(-1 * discrete['CTD Depth [m]'], discrete['Start Latitude [degrees]'])
    discrete['CTD Pressure [db]'] = discrete['CTD Pressure [db]'].fillna(dbar)
    
    # Fill in missing analysis temperatures with the average
    discrete['pCO2 Analysis Temp [deg C]'] = discrete['pCO2 Analysis Temp [deg C]'].fillna(discrete['pCO2 Analysis Temp [deg C]'].mean(skipna=True))
    
    # NaN the CTD and discrete sample data that have fail flags set
    discrete['CTD Pressure [db]'] = discrete['CTD Pressure [db]'].where(discrete['CTD Pressure Flag'] != 4)
    discrete['CTD Temperature 1 [deg C]'] = discrete['CTD Temperature 1 [deg C]'].where(discrete['CTD Temperature 1 Flag'] != 4)
    discrete['CTD Temperature 2 [deg C]'] = discrete['CTD Temperature 2 [deg C]'].where(discrete['CTD Temperature 2 Flag'] != 4)
    discrete['CTD Salinity 1 [psu]'] = discrete['CTD Salinity 1 [psu]'].where(discrete['CTD Conductivity 1 Flag'] != 4)
    discrete['CTD Salinity 2 [psu]'] = discrete['CTD Salinity 2 [psu]'].where(discrete['CTD Conductivity 2 Flag'] != 4)
    discrete['Discrete DIC [umol/kg]'] = discrete['Discrete DIC [umol/kg]'].where(discrete['Discrete DIC Flag'] != 4)
    discrete['Discrete pCO2 [uatm]'] = discrete['Discrete pCO2 [uatm]'].where(discrete['Discrete DIC Flag'] != 4)
    discrete['Calculated Alkalinity [umol/kg]'] = discrete['Calculated Alkalinity [umol/kg]'].where(discrete['Discrete DIC Flag'] != 4)
    
    # add the potential density to the dataframe
    psu = discrete[['CTD Salinity 1 [psu]', 'CTD Salinity 2 [psu]']].mean(skipna=True, axis=1)
    degC = discrete[['CTD Temperature 1 [deg C]', 'CTD Temperature 2 [deg C]']].mean(skipna=True, axis=1)
    dbar = discrete['CTD Pressure [db]']
    SA = SA_from_SP(psu, dbar, discrete['CTD Longitude [deg]'], discrete['CTD Latitude [deg]'])
    discrete['CTD Potential Density [kg/m^3]'] = pot_rho_t_exact(SA, degC, dbar, 0)

    # save the results
    return discrete


def load_ndbc(station):
    """
    Use the NDBC ERDDAP server to load NDBC data for station(s) that are
    close to the OOI Endurance Array moorings.

    :param station: NDBC station number
    :return ndbc: xarray dataset with the NDBC station data
    """
    # set up the basic request for the data from the NDBC ERDDAP server
    server = "https://erddap.aoml.noaa.gov/hdb/erddap"
    e = ERDDAP(
        server=server,
        protocol="tabledap",
        response="csv"
    )
    e.dataset_id = "NDBC_BUOY_1997_present"
    e.variables = ["time", "wspu", "wspv", "bar", "atmp", "wtmp"]
    drop = ['station', 'longitude', 'latitude', 'rowSize']
    
    # Set the request constraints and get the data
    e.constraints = {
        "station=": station,
        "time>=": "2015-04-01T00:00:00Z"
    }
    ndbc = e.to_xarray(requests_kwargs = {'verify': False}).squeeze()
    ndbc = ndbc.swap_dims({'obs': 'time'})
    for v in ndbc.variables:
        if v in drop:
            ndbc = ndbc.drop_vars(v)

    # sort the data by time, removing any duplicate timestamps
    ndbc = ndbc.squeeze()
    _, index = np.unique(ndbc['time'], return_index=True)
    ndbc = ndbc.isel(time=index)
    ndbc = ndbc.sortby('time')    
    return ndbc


def metadata_reset(ds, site, node, sensor, depth, attributes):
    """
    Update dataset metadata and set CF-compliant coordinates for a single time
    series at a fixed spatial location.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to update.
    site : str
        OOI site designator (part of the OOI reference designator).
    node : str
        OOI node designator (part of the OOI reference designator).
    sensor : str
        OOI sensor designator (part of the OOI reference designator).
    depth : sequence of float
        Sequence of three elements: deployment depth, vertical minimum, and
        vertical maximum extent for the instrument in this dataset.
    attributes : dict
        Global and variable-level attributes to apply to the dataset.

    Returns
    -------
    xarray.Dataset
        The updated dataset with the metadata applied.
    """
    # convert the depth array to named variables    # convert the depth array to named variables
    deploy_depth = depth[0]     # instrument deployment depth
    min_depth = depth[1]        # minimum vertical extent of the instrument depth
    max_depth = depth[2]        # maximum vertical extent of the instrument depth
    # note, the minimum and maximum depths will vary if the instrument includes a
    # pressure sensor, otherwise they will be set to the deployment depth

    # query the OOI M2M system for the deployment history of the mooring
    deployments = list_deployments(site, node, sensor)

    # use the listing of deployments to get the location history
    latitudes = []
    longitudes = []
    for deploy in deployments:
        data = get_sensor_information(site, node, sensor, deploy)
        latitudes.append(data[0]['location']['latitude'])
        longitudes.append(data[0]['location']['longitude'])
    
    # convert the location lists to arrays
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)

    # add the geospatial coordinates as scalar variables
    ds = ds.assign(longitude=longitudes.mean())
    ds = ds.assign(latitude=latitudes.mean())
    ds = ds.assign(depth=deploy_depth)
    ds = ds.assign(station_name=site)

    # create the attributes for the geospatial coordinates
    geo_coords = ['time', 'longitude', 'latitude', 'depth', 'station_name'] 
    geo_attrs = dict({
        'time': {
            'long_name': 'Time',
            'standard_name': 'time',
            # units are set below via the encoding so xarray saves them correctly
            'axis': 'T',
            'comment': ('Time record for the dataset. Note, this record may not be monotonic if the data from '
                        'overlapping deployments is preserved to allow users to utilize the overlaps in their '
                        'analysis (not the case for all datasets). Users will need to account for this fact when '
                        'analyzing the data. The deployment variable, if applicable, can be used to separate the '
                        'data into unique deployments. The data is monotonic on a per deployment basis or if the '
                        'dataset has been reworked to be a continuous timeseries.')
        },
        'longitude': {
            'long_name': 'Longitude',
            'standard_name': 'longitude',
            'units': 'degrees_east',
            'axis': 'X',
            'comment': 'Average deployment longitude for all deployments of this sensor.'
        },
        'latitude': {
            'long_name': 'Latitude',
            'standard_name': 'latitude',
            'units': 'degrees_north',
            'axis': 'Y',
            'comment': 'Average deployment latitude for all deployments of this sensor.'
        },
        'depth': {
            'long_name': 'Depth',
            'standard_name': 'depth',
            'units': 'm',
            'comment': ('Depth of the instrument, either from the deployment depth (e.g. 7 m for an NSIF) or the '
                        'average depth calculated from the instrument pressure record.'),
            'positive': 'down',
            'axis': 'Z'
        },
        'station_name': {
            'long_name': 'Station Name',
            'standard_name': 'platform_name',
            'cf_role': 'timeseries_id'
        }        
    })
    for v in geo_coords:
        # update the attributes for the geo coordinates
        ds[v].attrs = geo_attrs[v]
    
    # update the global attributes with deployment specific details
    time_start = ds['time'].min().dt.strftime('%Y-%m-%dT%H:%M:00Z').values
    time_end = ds['time'].max().dt.strftime('%Y-%m-%dT%H:%M:00Z').values
    time_now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:00Z")
    attributes = dict_update(attributes, SHARED)  # add the shared, global attributes
    attributes['global'] = dict_update(attributes['global'], {
        'ooi_reference_designator': f'{site}-{node}-{sensor}',
        'metadata_link': f'https://oceanobservatories.org/site/{site}/',
        'history': f'{time_now}: dataset created',
        'date_created': time_now,
        'date_released': time_now,
        'date_modified': time_now,
        'time_coverage_start': time_start,
        'time_coverage_end': time_end,
        'geospatial_lat_min': latitudes.min(),
        'geospatial_lat_max': latitudes.max(),
        'geospatial_lat_units': 'degrees_east',
        'geospatial_lon_min': longitudes.min(),
        'geospatial_lon_max': longitudes.max(),
        'geospatial_lon_units': 'degrees_north',
        'geospatial_vertical_min': min_depth,
        'geospatial_vertical_max': max_depth,
        'geospatial_vertical_positive': 'down',
        'geospatial_vertical_units': 'm'
    })
    
    # assign the updated attributes to the global metadata and the individual variables
    ds.attrs = attributes['global']
    for v in ds.variables:
        if v == 'deployment':
            ds[v].attrs = dict({
                'long_name': 'Deployment ID',
                'units': 'counts',
                'comment': ('Mooring deployment ID number. Useful for differentiating data by deployment, '
                            'allowing for overlapping deployments in the data sets.'),
                'coordinates': 'time longitude latitude depth station_name'
            })
        elif v not in geo_coords:
            ds[v].attrs = dict_update(attributes[v], {'coordinates': 'time longitude latitude depth station_name'})
        else:
            continue  # set up above as a geo coordinate above

    # add final encodings for the geo coordinate variables so xarray saves them correctly in the NetCDF files
    ds['time'].encoding = dict({
        '_FillValue': None,
        'units': 'seconds since 1900-01-01T00:00:00.000Z',
        'calendar': 'standard',
        'dtype': 'float64'
    })
    ds['longitude'].encoding = dict({'_FillValue': None})
    ds['latitude'].encoding = dict({'_FillValue': None})
    ds['depth'].encoding = dict({'_FillValue': None})
    ds['station_name'].encoding = dict({'dtype': 'S8'})
    
    # return the data set for further work
    return ds


def robust_regression(x, y):
    """
    Perform robust linear regression using RANSAC algorithm.

    Uses Random Sample Consensus (RANSAC) regression to fit a linear model
    while being resistant to outliers in the data.

    Parameters
    ----------
    x : array_like
        Independent variable data.
    y : array_like
        Dependent variable data.

    Returns
    -------
    slope : float
        Slope of the fitted line.
    intercept : float
        Y-intercept of the fitted line.
    r_squared : float
        Coefficient of determination (R²) for the fit.

    Notes
    -----
    NaN values are automatically removed before fitting. The RANSAC algorithm
    uses a minimum sample size of 10% of the data and a maximum of 2000 trials.
    """
    # mask (remove) any NaNs in the data
    mask = (np.isfinite(x) & np.isfinite(y))
    x = x[mask]
    y = y[mask]
    X = x[:, np.newaxis]

    # create the RANSAC regression object
    model = RANSACRegressor(LinearRegression(), min_samples=int(len(x)/10), max_trials=2000)
    model.fit(X, y)

    # pull out the slope, offset and r-squared values
    intercept = model.estimator_.intercept_
    slope = model.estimator_.coef_
    r_squared = model.estimator_.score(X, y)
    return slope[0], intercept, r_squared


def odr_regression(x, y, iqr=True):
    """
    Perform Type II regression using Orthogonal Distance Regression (ODR).

    Calculates a reduced major axis (RMA) regression that accounts for errors
    in both x and y variables, unlike ordinary least squares regression.

    Parameters
    ----------
    x : array_like
        Independent variable data.
    y : array_like
        Dependent variable data.
    iqr : bool, optional
        If True, remove outliers using the interquartile range (IQR) method
        before fitting (default is True).

    Returns
    -------
    scipy.odr.odrpack.Output
        ODR output object containing regression results including beta (fitted
        parameters), sd_beta (standard deviations), and other fit statistics.

    Notes
    -----
    NaN values are automatically removed before fitting. When `iqr=True`,
    outliers beyond 1.5 times the IQR from the quartiles are removed from
    both x and y data before regression.
    """
    # mask (remove) any NaNs in the data
    mask = (np.isfinite(x) & np.isfinite(y))
    x = x[mask]
    y = y[mask]

    if iqr:
        # identify potential outliers in both the x and y arrays using the IQR method
        q1 = np.percentile(x, 25, method='midpoint')
        q3 = np.percentile(x, 75, method='midpoint')
        iqr = q3 - q1
        upper_x = q3 + 1.5 * iqr
        lower_x = q1 - 1.5 * iqr

        q1 = np.percentile(y, 25, method='midpoint')
        q3 = np.percentile(y, 75, method='midpoint')
        iqr = q3 - q1
        upper_y = q3 + 1.5 * iqr
        lower_y = q1 - 1.5 * iqr

        mask = (((x >= lower_x) & (x <= upper_x)) & ((y >= lower_y) & (y <= upper_y)))
        x = x[mask]
        y = y[mask]

    # calculate the regression using the ODR method
    linear = odr.polynomial(1)  # 1st-order polynomial
    data = odr.Data(x, y, wd=1./np.power(x.std(), 2), we=1./np.power(y.std(), 2))
    od = odr.ODR(data, linear, beta0=[0., 1.])
    res = od.run()
    return res
