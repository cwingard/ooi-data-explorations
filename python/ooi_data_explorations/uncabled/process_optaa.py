#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import re
import sys
import time
import xarray as xr

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from ooi_data_explorations.common import inputs, get_vocabulary, m2m_request, list_files, m2m_collect, \
    load_gc_thredds, update_dataset, N_CORES, ENCODINGS, FILL_INT
from ooi_data_explorations.profilers import create_profile_id, bin_profiles
from ooi_data_explorations.uncabled.utilities.utilities_optaa import load_cal_coefficients, apply_dev, apply_tscorr, \
    apply_scatcorr, estimate_chl_poc, calculate_ratios, PureWater, tscor
from pyseas.data.opt_functions import opt_internal_temp, opt_external_temp

# reset the variable level attributes and set some global defaults
ATTRS = dict({
    # parsed (raw) variables and attributes
    'serial_number': {
        'long_name': 'Unit Serial Number',
        # 'units': ''    # deliberately left blank, no units for this value
        'comment': ('Instrument serial number, useful for tracking history of instrument through deployments and '
                    'to make sure the correct calibration coefficients have been applied.')
    },
    'external_temp_raw': {
        'long_name': 'Raw In-Situ Temperature',
        'units': 'count',
        'data_product_identifier': 'OPTTEMP_L0',
        'comment': ('Raw measurements, reported in counts, from the AC-S external temperature sensor. This sensor '
                    'measures the in-situ seawater temperature.')
    },
    'internal_temp_raw': {
        'long_name': 'Raw Internal Instrument Temperature',
        'units': 'count',
        'comment': ('Raw measurements, reported in counts, from the AC-S internal temperature sensor. This sensor '
                    'measures the internal instrument temperature and is used in converting the raw optical '
                    'measurements into absorbance and attenuation estimates.')
    },
    'elapsed_run_time': {
        'long_name': 'Elapsed Run Time',
        'units': 'ms',
        'comment': 'Time in milliseconds since the instrument was powered on.'
    },
    'wavelength_number': {
        'long_name': 'Wavelength Number',
        'units': 'count',
        'comment': ('An index counter between 0 and 99 used to set a common length dimension for the absorbance and '
                    'attenuation measurements. The actual number of wavelengths is variable between sensors '
                    'and may even change for a particular sensor over time if servicing requires a replacement '
                    'of the filter set. The actual number of wavelengths for this sensor is represented here '
                    'by the attribute actual_wavelengths.')
        # 'actual_wavelengths': ''  # deliberately left blank, created during the processing
    },
    'wavelength_a': {
        'long_name': 'Absorption Channel Wavelengths',
        'standard_name': 'radiation_wavelength',
        'units': 'nm',
        'comment': ('Absorbance channel measurement wavelengths, specific to the filter wheel set installed in '
                    'the AC-S.'),
        '_FillValue': np.nan
    },
    'a_reference_dark': {
        'long_name': 'Absorption Channel Dark Reference',
        'units': 'count',
        'comment': ('Absorption channel reference detector dark counts (before the lamp is turned on). Used in '
                    'conversion of the raw absorption channel measurements to absorbance estimates.')
    },
    'a_reference': {
        'long_name': 'Absorption Channel Reference Measurements',
        'units': 'count',
        'comment': ('Absorption channel reference detector raw counts (while the lamp is turned on). Used in '
                    'conversion of the raw absorption channel measurements to absorbance estimates.'),
        'data_product_identifier': 'OPTAREF_L0',
        '_FillValue': FILL_INT
    },
    'a_signal_dark': {
        'long_name': 'Absorption Channel Dark Signal',
        'units': 'count',
        'comment': ('Absorption channel signal detector dark counts (before the lamp is turned on). Used in conversion '
                    'of the raw absorption channel measurements to absorbance estimates.')
    },
    'a_signal': {
        'long_name': 'Absorption Channel Signal Measurements',
        'units': 'count',
        'comment': ('Absorption channel signal detector raw counts (while the lamp is turned on). Used in conversion '
                    'of the raw absorption channel measurements to absorbance estimates.'),
        'data_product_identifier': 'OPTASIG_L0',
        '_FillValue': FILL_INT
    },
    'wavelength_c': {
        'long_name': 'Attenuation Channel Wavelengths',
        'standard_name': 'radiation_wavelength',
        'units': 'nm',
        'comment': ('Attenuation channel measurement wavelengths, specific to the filter wheel set installed in '
                    'the AC-S.'),
        '_FillValue': np.nan
    },
    'c_reference_dark': {
        'long_name': 'Attenuation Channel Dark Reference',
        'units': 'count',
        'comment': ('Attenuation channel reference detector dark counts (before the lamp is turned on). Used in '
                    'conversion of the raw attenuation channel measurements to attenuation estimates.')
    },
    'c_reference': {
        'long_name': 'Attenuation Channel Reference Measurements',
        'units': 'count',
        'comment': ('Attenuation channel reference detector raw counts (while the lamp is turned on). Used in '
                    'conversion of the raw attenuation channel measurements to attenuation estimates.'),
        'data_product_identifier': 'OPTCREF_L0',
        '_FillValue': FILL_INT
    },
    'c_signal_dark': {
        'long_name': 'Attenuation Channel Dark Signal',
        'units': 'count',
        'comment': ('Attenuation channel signal detector dark counts (before the lamp is turned on). Used in '
                    'conversion of the raw attenuation channel measurements to attenuation estimates.')
    },
    'c_signal': {
        'long_name': 'Attenuation Channel Signal Measurements',
        'units': 'count',
        'comment': ('Attenuation channel signal detector raw counts (while the lamp is turned on). Used in conversion '
                    'of the raw attenuation channel measurements to attenuation estimates.'),
        'data_product_identifier': 'OPTCSIG_L0',
        '_FillValue': FILL_INT
    },

    # Data from a co-located CTD, if available, interpolated into the data set
    'sea_water_temperature': {
        'long_name': 'Seawater Temperature',
        'standard_name': 'sea_water_temperature',
        'units': 'degree_Celsius',
        'comment': ('Sea water temperature is the in situ temperature of the sea water near the sensor. Measurements '
                    'are from a co-located CTD'),
        'data_product_identifier': 'TEMPWAT_L1',
        '_FillValue': np.nan
    },
    'sea_water_practical_salinity': {
        'long_name': 'Practical Salinity',
        'standard_name': 'sea_water_practical_salinity',
        'units': '1',
        'comment': ('Salinity is generally defined as the concentration of dissolved salt in a parcel of sea water. '
                    'Practical Salinity is a more specific unit-less quantity calculated from the conductivity of '
                    'sea water and adjusted for temperature and pressure. It is approximately equivalent to Absolute '
                    'Salinity (the mass fraction of dissolved salt in sea water), but they are not interchangeable. '
                    'Measurements are from a co-located CTD.'),
        'data_product_identifier': 'PRACSAL_L2',
        '_FillValue': np.nan
    },
    'sea_water_pressure': {
        'long_name': 'Seawater Pressure',
        'standard_name': 'sea_water_pressure',
        'units': 'dbar',
        'comment': ('Seawater Pressure refers to the pressure exerted on a sensor in situ by the weight of the column '
                    'of seawater above it. It is calculated by subtracting one standard atmosphere from the absolute '
                    'pressure at the sensor to remove the weight of the atmosphere on top of the water column. The '
                    'pressure at a sensor in situ provides a metric of the depth of that sensor. Measurements are from '
                    'a co-located CTD'),
        'data_product_identifier': 'PRESWAT_L1',
        '_FillValue': np.nan
    },

    # OOI calculated beam attenuation and optical absorption
    'beam_attenuation': {
        'long_name': 'Optical Beam Attenuation Coefficient',
        'units': 'm-1',
        'comment': ('The Optical Beam Attenuation Coefficient is the rate that the intensity of a beam of light will '
                    'decrease in response to the combined effects of absorption and scatter as a function of '
                    'propagation distance. The Attenuation Coefficient results from the spectral beam attenuation of '
                    'the combination of all seawater impurities including all particulate and dissolved matter of '
                    'optical importance.'),
        'data_product_identifier': 'OPTATTN_L2',
        'ancillary_variables': ('wavelength_c internal_temp c_signal c_reference c_signal_dark c_reference_dark '
                                'seawater_temperature practical_salinity'),
        '_FillValue': np.nan
    },
    'optical_absorption': {
        'long_name': 'Optical Absorption Coefficient',
        'standard_name': 'volume_absorption_coefficient_of_radiative_flux_in_sea_water',
        'units': 'm-1',
        'comment': ('Optical Absorption Coefficient is the rate that the intensity of a beam of light will decrease '
                    'in response to the absorption (removal) of light energy as a function of propagation distance. '
                    'The Optical Absorption Coefficient reflects the absorption coefficient for the combination of all '
                    'seawater impurities including all particulate and dissolved matter of optical importance.'),
        'data_product_identifier': 'OPTABSN_L2',
        'ancillary_variables': ('wavelength_a internal_temp a_signal a_reference a_signal_dark a_reference_dark '
                                'seawater_temperature practical_salinity'),
        '_FillValue': np.nan
    },

    # Derived values in the re-processed data set
    'external_temp': {
        'long_name': 'External Instrument Temperature',
        'standard_name': 'sea_water_temperature',
        'units': 'degrees_Celsius',
        'comment': ('In-situ sea water temperature measurements from the sensor mounted at the top of the '
                    'AC-S pressure housing.'),
        'ancillary_variables': 'external_temp_raw'
    },
    'internal_temp': {
        'long_name': 'Internal Instrument Temperature',
        'units': 'degrees_Celsius',
        'comment': 'Internal instrument temperature, used to convert raw absorbance and attenuation measurements.',
        'ancillary_variables': 'internal_temp_raw'
    },
    'apg': {
        'long_name': 'Particulate and Dissolved Absorbance',
        'units': 'm-1',
        'comment': ('The optical absorption coefficient is the rate that the intensity of a beam of light will '
                    'decrease in response to the absorption (removal) of light energy as a function of propagation '
                    'distance. The optical absorption coefficient reflects the absorption coefficient for the '
                    'combination of all seawater impurities, including all particulate and dissolved matter of '
                    'optical importance.'),
        'data_product_identifier': 'OPTABSN_L1',
        'ancillary_variables': ('wavelength_a internal_temp a_signal_raw a_reference_raw '
                                'a_signal_dark a_reference_dark'),
        '_FillValue': np.nan
    },
    'a_jump_offsets': {
        'long_name': 'Absorption Channel Holographic Grater Jump Offset',
        'units': 'm-1',
        'comment': ('Offset used to correct for spectral jumps commonly seen in the AC-S data where the sensor uses '
                    'two holographic gratings to span the full spectral range. Adding the offset to all values '
                    'from the grate_index (included as an additional attribute) to the end of the spectra will '
                    'restore the AC-S data to values reported by the sensor.'),
        'grate_index': 0,  # Will update in the processing script
        'ancillary_variables': 'wavelength_a apg',
    },
    'apg_ts': {
        'long_name': 'Particulate and Dissolved Absorbance with TS Correction',
        'units': 'm-1',
        'comment': ('The optical absorption coefficient corrected for the effects of temperature and salinity. '
                    'Utilizes data from a co-located CTD for the temperature and salinity, if available. If no '
                    'co-located CTD data is available, will assume a constant salinity of 33 psu and will use '
                    'the OPTAA''s external temperature sensor.'),
        'ancillary_variables': 'wavelength_a sea_water_temperature sea_water_practical_salinity external_temp apg',
        '_FillValue': np.nan
    },
    'apg_ts_s': {
        'long_name': 'Particulate and Dissolved Absorbance with TS and Scatter Correction',
        'standard_name': 'volume_absorption_coefficient_of_radiative_flux_in_sea_water',
        'units': 'm-1',
        'comment': ('The optical absorption coefficient corrected for scattering after correcting for temperature and '
                    'salinity. Utilizes Method 1 (baseline correction) by subtracting the absorption at 715 nm from '
                    'all values; assumes scattering is flat across wavelengths.'),
        'data_product_identifier': 'OPTABSN_L2',
        'ancillary_variables': 'wavelength_a apg_ts',
        '_FillValue': np.nan
    },
    'cpg': {
        'long_name': 'Particulate and Dissolved Attenuation',
        'units': 'm-1',
        'comment': ('The optical beam attenuation coefficient is the rate that the intensity of a beam of light will '
                    'decrease in response to the combined effects of absorption and scatter as a function of '
                    'propagation distance. The attenuation coefficient results from the spectral beam attenuation of '
                    'the combination of all seawater impurities including all particulate and dissolved matter of '
                    'optical importance.'),
        'data_product_identifier': 'OPTATTN_L1',
        'ancillary_variables': ('wavelength_c internal_temp c_signal_raw c_reference_raw '
                                'c_signal_dark c_reference_dark'),
        '_FillValue': np.nan
    },
    'c_jump_offsets': {
        'long_name': 'Attenuation Channel Filter Offsets',
        'units': 'm-1',
        'comment': ('Offset used to correct for spectral jumps commonly seen in the AC-S data where the sensor uses '
                    'two holographic gratings to span the full spectral range. Adding the offset to all values '
                    'from the grate_index (included as an additional attribute) to the end of the spectra will '
                    'restore the AC-S data to values reported by the sensor.'),
        'grate_index': 0,  # Will update in the processing script
        'ancillary_variables': 'wavelength_c cpg',
    },
    'cpg_ts': {
        'long_name': 'Particulate and Dissolved Attenuation with TS Correction',
        'units': 'm-1',
        'comment': ('The optical beam attenuation coefficient corrected for the effects of temperature and salinity. '
                    'Utilizes data from a co-located CTD for the temperature and salinity, if available. If no '
                    'co-located CTD data is available, will assume a constant salinity of 33 psu and will use '
                    'the OPTAA''s external temperature sensor.'),
        'data_product_identifier': 'OPTATTN_L2',
        'ancillary_variables': 'wavelength_c sea_water_temperature sea_water_practical_salinity external_temp cpg',
        '_FillValue': np.nan
    },
    'estimated_chlorophyll': {
        'long_name': 'Estimated Chlorophyll Concentration',
        'standard_name': 'mass_concentration_of_chlorophyll_in_sea_water',
        'units': 'ug L-1',
        'comment': ('Uses the absorption line height at 676 nm, above a linear background between 650 and 715 nm, with '
                    'a chlorophyll specific absorption of 0.020 L/ug/m to estimate the concentration of chlorophyll. '
                    'This method has been shown to be significantly related to extracted chlorophyll concentrations '
                    'and is robust in response to mild to moderate bio-fouling.'),
        'ancillary_variables': 'apg_ts_s',
        '_FillValue': np.nan
    },
    'estimated_poc': {
        'long_name': 'Estimated POC Concentration',
        'standard_name': 'mass_concentration_of_organic_detritus_expressed_as_carbon_in_sea_water',
        'units': 'ug L-1',
        'comment': ('Uses the particulate beam attenuation coefficient at 660 nm and a coefficient of 380 ug/L/m. This '
                    'calculation is not robust in response to bio-fouling and is expected to breakdown as bio-fouling '
                    'begins to dominate the signal.'),
        'ancillary_variables': 'cpg_ts',
        '_FillValue': np.nan
    },
    'ratio_cdom': {
        'long_name': 'CDOM to Chlorophyll Absorbance Ratio',
        'units': '1',
        'comment': ('Ratio of CDOM absorption in the violet portion of the spectrum at 412 nm relative to '
                    'chlorophyll absorption at 440 nm. Ratios greater than 1 indicate a preponderance of CDOM '
                    'absorption relative to chlorophyll.'),
        'ancillary_variables': 'apg_ts_s',
        '_FillValue': np.nan
    },
    'ratio_carotenoids': {
        'long_name': 'Carotenoid to Chlorophyll Absorbance Ratio',
        'units': '1',
        'comment': ('Ratio of carotenoid absorption in the blue-green portion of the spectrum at 490 nm relative to '
                    'chlorophyll absorption at 440 nm. A changing carotenoid to chlorophyll ratio may indicate a shift '
                    'in phytoplankton community composition in addition to changes in light history or bloom health '
                    'and age.'),
        'ancillary_variables': 'apg_ts_s',
        '_FillValue': np.nan
    },
    'ratio_phycobilins': {
        'long_name': 'Phycobilins to Chlorophyll Absorbance Ratio',
        'units': '1',
        'comment': ('Ratio of phycobilin absorption in the green portion of the spectrum at 530 nm relative to '
                    'chlorophyll absorption at 440 nm. Different phytoplankton, notably cyanobacteria, utilize '
                    'phycobilins as accessory light harvesting pigments. An increasing phycobilin to chlorophyll ratio '
                    'may indicate a shift in phytoplankton community composition.'),
        'ancillary_variables': 'apg_ts_s',
        '_FillValue': np.nan
    },
    'ratio_qband': {
        'long_name': 'Chlorophyll Q Band to Soret Band Absorbance Ratio',
        'units': '1',
        'comment': ('The Soret and the Q bands represent the two main absorption bands of chlorophyll. The former '
                    'covers absorption in the blue region of the spectrum, while the latter covers absorption in the '
                    'red region. A decrease in the ratio of the intensity of the Soret band at 440 nm to that of the Q '
                    'band at 676 nm may indicate a change in phytoplankton community structure. All phytoplankton '
                    'contain chlorophyll a as the primary light harvesting pigment, but green algae and '
                    'dinoflagellates contain chlorophyll b and c, respectively, which are spectrally redshifted '
                    'compared to chlorophyll a.'),
        'ancillary_variables': 'apg_ts_s',
        '_FillValue': np.nan
    }
})


def optaa_datalogger(ds, cal_file, a_purewater_file = None, c_purewater_file=None):
    """
    Takes OPTAA data recorded by the data loggers used in the CGSN/EA moorings
    and cleans up the data set to make it more user-friendly.  Primary task is
    renaming parameters and dropping some that are of limited use. Additionally,
    re-calculate the intermediate products (e.g. absorption and attenuation) and
    add them to the data set.  Finally, add the estimated chlorophyll and POC
    concentrations to the data set.

    Will test the data set to determine if more than one deployment is present.
    If so, will raise an exception with an error message.  AC-S processing
    requires that the data be processed one deployment at a time in order to
    properly assign calibration coefficients.

    :param ds: initial optaa data set downloaded from OOI via the M2M system
    :param cal_file: file name (can include path) to store the calibration
        coefficients
    :param a_purewater_file: file name with path of the purewater calibration
        file for the a-channel. If left None will not apply purewater correction
    :param c_purewater_file: file name with path of the purewater calibration
        file for the c-channel. If left None will not apply purewater correction
    :return ds: cleaned up data set
    """
    # check to see if there is more than one deployment in the data set
    if len(np.unique(ds['deployment'].values)) > 1:
        raise ValueError('More than one deployment in the data set.  Please structure processing request to process '
                         'one deployment at a time.')

    # drop some of the variables:
    #   internal_timestamp == time, redundant so can remove
    #   pressure_counts == none of the OOI OPTAAs have a pressure sensor
    ds = ds.drop_vars(['internal_timestamp', 'pressure_counts'])

    # check for data from a co-located CTD, if not present create the variables using NaN's as the fill value
    if 'sea_water_temperature' not in ds.variables:
        ds['sea_water_temperature'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_practical_salinity'] = ('time', ds['deployment'].data * np.nan)

    # pull out the number of wavelengths and serial number and then drop the variable (part of the metadata)
    num_wavelengths = ds.num_wavelengths.values[0].astype(int)
    serial_number = int(re.sub('[^0-9]', '', ds.attrs['SerialNumber']))
    ds = ds.drop_vars('num_wavelengths')

    # load the calibration coefficients
    uid = ds.attrs['AssetUniqueID']
    start_time = ds['time'][0].values.astype(float) / 10 ** 9
    cal = load_cal_coefficients(cal_file, uid, start_time)

    # check the calibration coefficients against the deployment data
    if cal.coeffs['serial_number'] != serial_number:
        raise Exception('Serial Number mismatch between ac-s data and the device file.')
    if cal.coeffs['num_wavelengths'] != num_wavelengths:
        raise Exception('Number of wavelengths mismatch between ac-s data and the device file.')

    # remove the units from the variable names
    rename = {
        'a_signal_dark_counts': 'a_signal_dark',
        'a_reference_dark_counts': 'a_reference_dark',
        'a_signal_counts': 'a_signal',
        'a_reference_counts': 'a_reference',
        'c_signal_dark_counts': 'c_signal_dark',
        'c_reference_dark_counts': 'c_reference_dark',
        'c_signal_counts': 'c_signal',
        'c_reference_counts': 'c_reference',
        'wavelength': 'wavelength_number'
    }
    ds = ds.rename(rename)

    # Delete the first 45 seconds of the data record per recommendation from the vendor. Note, originally the vendor
    # recommended deleting the first 45 seconds, then 60 seconds and then 120 seconds.  They never provided a data
    # based reason for the change in recommendation. Within OOI, instruments were programmed to run for 60 seconds,
    # then 120 seconds and then 240 seconds ... and it is all mixed up across the various data sets.  So, we are
    # going to use the 45-second recommendation and apply it to all data sets. If the vendor ever provides an analysis
    # justifying the change in recommendation, we can revisit this.
    ds.elapsed_run_time.values = ds.elapsed_run_time.where(ds.elapsed_run_time / 1000 > 45)
    ds = ds.dropna(dim='time', subset=['elapsed_run_time'])

    # convert internal and external temperature sensors from raw counts to degrees Celsius
    ds['internal_temp'] = opt_internal_temp(ds['internal_temp_raw'])
    ds['external_temp'] = opt_external_temp(ds['external_temp_raw'])

    # calculate the median of the remaining data per burst measurement
    print('Calculating burst averages ...')
    start_time = time.time()
    ds['time'] = ds['time'] + np.timedelta64(450, 's')
    burst = ds.resample(time='900s', skipna=True).reduce(np.median, dim='time', keep_attrs=True)
    burst = burst.where(~np.isnan(burst.deployment), drop=True)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print('... burst averaging complete.  Elapsed time: %f seconds' % elapsed_time)

    # re-process the raw data in order to create the intermediate variables, correcting for the holographic
    # grating, applying the temperature and salinity corrections and applying a baseline scatter correction
    # to the absorption data. All intermediate processing outputs are added to the data set.
    burst = apply_dev(burst, cal.coeffs)
    burst = apply_tscorr(burst, cal.coeffs, burst.sea_water_temperature, burst.sea_water_practical_salinity)
    if a_purewater_file is not None:
        # Apply the purewater correction to the a-channel
        purewater_a = PureWater(a_purewater_file, 'a', cal, tscor, ATTRS)
        burst["apg_ts"] = burst["apg_ts"] - purewater_a.dat["a_signal_ts"].median(dim="time")
    if c_purewater_file is not None:
        # Apply the purewater correction to the c-channel
        purewater_c = PureWater(c_purewater_file, 'c', cal, tscor, ATTRS)
        burst["cpg_ts"] = burst["cpg_ts"] - purewater_c.dat["c_signal_ts"].median(dim="time")
    burst = apply_scatcorr(burst, cal.coeffs)

    # add the jump offsets as NaN's if the grating index correction was not used
    if 'a_jump_offsets' not in ds.variables:
        ds['a_jump_offsets'] = ('time', ds['deployment'].data * np.nan)
        ds['c_jump_offsets'] = ('time', ds['deployment'].data * np.nan)

    # estimate chlorophyll and POC and calculate select absorption ratios
    burst = estimate_chl_poc(burst, cal.coeffs)
    burst = calculate_ratios(burst)

    # create a xarray dataset of the 2D variables, padding the number of wavelengths to a consistent
    # length of 100 using fill values.
    wavelength_number = np.arange(100).astype(int)  # used as a dimensional variable
    pad = 100 - num_wavelengths
    fill_nan = np.tile(np.ones(pad) * np.nan, (len(burst.time), 1))
    fill_int = np.tile(np.ones(pad) * FILL_INT, (len(burst.time), 1))

    wavelength_a = np.concatenate([burst.wavelength_a.values, fill_nan], axis=1)
    wavelength_c = np.concatenate([burst.wavelength_c.values, fill_nan], axis=1)

    ac = xr.Dataset({
        'wavelength_a': (['time', 'wavelength_number'], wavelength_a),
        'a_signal': (['time', 'wavelength_number'], np.concatenate([burst.a_signal, fill_int], axis=1).astype(int)),
        'a_reference': (['time', 'wavelength_number'], np.concatenate([burst.a_reference, fill_int],
                                                                      axis=1).astype(int)),
        'optical_absorption': (['time', 'wavelength_number'], np.concatenate([burst.optical_absorption, fill_nan],
                                                                             axis=1)),
        'apg': (['time', 'wavelength_number'], np.concatenate([burst.apg, fill_nan], axis=1)),
        'apg_ts': (['time', 'wavelength_number'], np.concatenate([burst.apg_ts, fill_nan], axis=1)),
        'apg_ts_s': (['time', 'wavelength_number'], np.concatenate([burst.apg_ts_s, fill_nan], axis=1)),
        'wavelength_c': (['time', 'wavelength_number'], wavelength_c),
        'c_signal': (['time', 'wavelength_number'], np.concatenate([burst.c_signal, fill_int], axis=1).astype(int)),
        'c_reference': (['time', 'wavelength_number'], np.concatenate([burst.c_reference, fill_int],
                                                                      axis=1).astype(int)),
        'beam_attenuation': (['time', 'wavelength_number'], np.concatenate([burst.beam_attenuation, fill_nan], axis=1)),
        'cpg': (['time', 'wavelength_number'], np.concatenate([burst.cpg, fill_nan], axis=1)),
        'cpg_ts': (['time', 'wavelength_number'], np.concatenate([burst.cpg_ts, fill_nan], axis=1)),
    }, coords={'time': (['time'], burst.time.values), 'wavelength_number': wavelength_number})

    # drop the original 2D variables from the burst data set
    drop = burst.drop_vars(['wavelength_number', 'wavelength_a', 'a_signal', 'a_reference',
                       'optical_absorption', 'apg', 'apg_ts', 'apg_ts_s',
                       'wavelength_c', 'c_signal', 'c_reference',
                       'beam_attenuation', 'cpg', 'cpg_ts'])

    # reset the data type for the 'a' and 'c' signal and reference dark values, and the other raw parameters
    int_arrays = ['a_signal_dark', 'a_reference_dark', 'c_signal_dark', 'c_reference_dark',
                  'internal_temp_raw', 'external_temp_raw', 'deployment']
    for k in drop.variables:
        if k in int_arrays:
            drop[k] = drop[k].astype(int)

    # recombine the two datasets
    optaa = xr.merge([drop, ac])

    # reset the attributes, which the merging drops
    optaa.attrs = burst.attrs
    for v in optaa.variables:
        optaa[v].attrs = burst[v].attrs

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in optaa.variables:
                optaa[key].attrs[atk] = atv

    # add the original variable name as an attribute, if renamed
    for key, value in rename.items():
        optaa[value].attrs['ooinet_variable_name'] = key

    # add the actual number of wavelengths to the dataset as an attribute
    optaa['wavelength_number'].attrs['actual_wavelengths'] = num_wavelengths

    # if the filter index was used to adjust the spectral jumps, add that attribute to the data set
    if cal.coeffs['grate_index']:
        optaa['a_jump_offsets'].attrs['grate_index'] = cal.coeffs['grate_index']
        optaa['c_jump_offsets'].attrs['grate_index'] = cal.coeffs['grate_index']

    return optaa


def optaa_cspp(ds, cal_file):
    """
    Takes OPTAA data recorded by the Coastal Surface-Piercing Profiler (CSPP)
    and cleans up the data set to make it more user-friendly.  Primary task is
    renaming parameters and dropping some that are of limited use.  Additionally,
    re-calculate the intermediate products (e.g. absorption and attenuation) and
    add them to the data set.  Finally, add the estimated chlorophyll and POC
    concentrations to the data set.

    Will test the data set to determine if more than one deployment is present.
    If so, will raise an exception with an error message.  AC-S processing
    requires that the data be processed one deployment at a time in order to
    properly assign calibration coefficients.

    :param ds: initial optaa data set downloaded from OOI via the M2M system
    :param cal_file: file name (can include path) to store the calibration
        coefficients
    :return ds: cleaned up data set
    """
    # check to see if there is more than one deployment in the data set
    if len(np.unique(ds['deployment'].values)) > 1:
        raise ValueError('More than one deployment in the data set.  Please structure processing request to process '
                         'one deployment at a time.')

    # drop some of the variables:
    #   internal_timestamp == time, redundant so can remove
    #   profiler_timestamp == internal_timestamp == time, redundant so can remove
    #   suspect_timestamp = not used
    #   pressure_counts == none of the OOI OPTAAs have a pressure sensor
    ds = ds.drop_vars(['internal_timestamp', 'profiler_timestamp', 'suspect_timestamp', 'pressure_counts'])

    # check for data from a co-located CTD, if not present create the variables using NaN's as the fill value
    if 'sea_water_temperature' not in ds.variables:
        ds['sea_water_temperature'] = ('time', ds['deployment'].data * np.nan)
        ds['sea_water_practical_salinity'] = ('time', ds['deployment'].data * np.nan)

    # pull out the number of wavelengths and serial number and then drop the variable (part of the metadata)
    num_wavelengths = ds.num_wavelengths.values[0].astype(int)
    serial_number = int(re.sub('[^0-9]', '', ds.attrs['SerialNumber']))
    ds = ds.drop_vars('num_wavelengths')

    # load the calibration coefficients
    uid = ds.attrs['AssetUniqueID']
    start_time = ds['time'][0].values.astype(float) / 10 ** 9
    cal = load_cal_coefficients(cal_file, uid, start_time)

    # check the calibration coefficients against the deployment data
    if cal.coeffs['serial_number'] != serial_number:
        raise Exception('Serial Number mismatch between ac-s data and the device file.')
    if cal.coeffs['num_wavelengths'] != num_wavelengths:
        raise Exception('Number of wavelengths mismatch between ac-s data and the device file.')

    # remove the units from the variable name
    rename = {
        'a_signal_dark_counts': 'a_signal_dark',
        'a_reference_dark_counts': 'a_reference_dark',
        'a_signal_counts': 'a_signal',
        'a_reference_counts': 'a_reference',
        'c_signal_dark_counts': 'c_signal_dark',
        'c_reference_dark_counts': 'c_reference_dark',
        'c_signal_counts': 'c_signal',
        'c_reference_counts': 'c_reference',
        'on_seconds': 'elapsed_run_time',
        'int_ctd_pressure': 'sea_water_pressure',
        'wavelength': 'wavelength_number'
    }
    ds = ds.rename(rename)

    # Delete the first 45 seconds of the data record per recommendation from the vendor. Note, originally the vendor
    # recommended deleting the first 45 seconds, then 60 seconds and then 120 seconds.  They never provided a data
    # based reason for the change in recommendation. Within OOI, instruments were programmed to run for 60 seconds,
    # then 120 seconds and then 240 seconds ... and it is all mixed up across the various data sets.  So, we are
    # going to use the 45-second recommendation and apply it to all data sets. If the vendor ever provides an analysis
    # justifying the change in recommendation, we can revisit this.
    ds['elapsed_run_time'] = ds['elapsed_run_time'] * 1000
    ds.elapsed_run_time.values = ds.elapsed_run_time.where(ds.elapsed_run_time / 1000 > 45)
    ds = ds.dropna(dim='time', subset=['elapsed_run_time'])

    # convert internal and external temperature sensors from raw counts to degrees Celsius
    ds['internal_temp'] = opt_internal_temp(ds['internal_temp_raw'])
    ds['external_temp'] = opt_external_temp(ds['external_temp_raw'])

    # create a profile variable to uniquely identify profiles within the dataset
    print('Creating and adding a profile variable to the data set ...')
    ds = create_profile_id(ds)

    # group the data by profile number and bin the data into 25 cm depth bins (nominal ascent rate of the CSPP)
    vocab = get_vocabulary(ds.attrs['subsite'], ds.attrs['node'], ds.attrs['sensor'])[0]
    site_depth = vocab['maxdepth'] - 2
    profiles = ds.groupby('profile')
    profiles = [profile[1] for profile in profiles]
    partial_binning = partial(bin_profiles, site_depth=site_depth, bin_size=0.25)
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        binned = list(tqdm(executor.map(partial_binning, profiles), total=len(profiles),
                           desc='Smoothing and binning each profile into 25 cm depth bins', file=sys.stdout))

    # reset the dataset now using binned profiles
    binned = [i for i in binned if i is not None]
    binned = xr.concat(binned, 'time')
    binned = binned.sortby(['profile', 'time'])

    # confirm dimension order is correct for the wavelength arrays (sometimes the order gets flipped
    # during the binning process)
    binned['wavelength_a'] = binned.wavelength_a.transpose(*['time', 'wavelength_number'])
    binned['wavelength_c'] = binned.wavelength_c.transpose(*['time', 'wavelength_number'])

    # re-process the raw data in order to create the intermediate variables, correcting for the holographic
    # grating, applying the temperature and salinity corrections and applying a baseline scatter correction
    # to the absorption data. All intermediate processing outputs are added to the data set.
    binned = apply_dev(binned, cal.coeffs)
    binned = apply_tscorr(binned, cal.coeffs, binned.sea_water_temperature, binned.sea_water_practical_salinity)
    binned = apply_scatcorr(binned, cal.coeffs)

    # add the jump offsets as NaN's if the grating index correction was not used
    if 'a_jump_offsets' not in ds.variables:
        ds['a_jump_offsets'] = ('time', ds['deployment'].data * np.nan)
        ds['c_jump_offsets'] = ('time', ds['deployment'].data * np.nan)

    # estimate chlorophyll and POC and calculate select absorption ratios
    binned = estimate_chl_poc(binned, cal.coeffs)
    binned = calculate_ratios(binned)

    # create a xarray dataset of the 2D variables, padding the number of wavelengths to a consistent
    # length of 100 using fill values.
    wavelength_number = np.arange(100).astype(int)  # used as a dimensional variable
    pad = 100 - num_wavelengths
    fill_nan = np.tile(np.ones(pad) * np.nan, (len(binned.time), 1))
    fill_int = np.tile(np.ones(pad) * FILL_INT, (len(binned.time), 1))

    wavelength_a = np.concatenate([binned.wavelength_a.values, fill_nan], axis=1)
    wavelength_c = np.concatenate([binned.wavelength_c.values, fill_nan], axis=1)

    ac = xr.Dataset({
        'wavelength_a': (['time', 'wavelength_number'], wavelength_a),
        'a_signal': (['time', 'wavelength_number'], np.concatenate([binned.a_signal, fill_int], axis=1).astype(int)),
        'a_reference': (['time', 'wavelength_number'], np.concatenate([binned.a_reference, fill_int],
                                                                      axis=1).astype(int)),
        'optical_absorption': (['time', 'wavelength_number'], np.concatenate([binned.optical_absorption, fill_nan],
                                                                             axis=1)),
        'apg': (['time', 'wavelength_number'], np.concatenate([binned.apg, fill_nan], axis=1)),
        'apg_ts': (['time', 'wavelength_number'], np.concatenate([binned.apg_ts, fill_nan], axis=1)),
        'apg_ts_s': (['time', 'wavelength_number'], np.concatenate([binned.apg_ts_s, fill_nan], axis=1)),
        'wavelength_c': (['time', 'wavelength_number'], wavelength_c),
        'c_signal': (['time', 'wavelength_number'], np.concatenate([binned.c_signal, fill_int], axis=1).astype(int)),
        'c_reference': (['time', 'wavelength_number'], np.concatenate([binned.c_reference, fill_int],
                                                                      axis=1).astype(int)),
        'beam_attenuation': (['time', 'wavelength_number'], np.concatenate([binned.beam_attenuation, fill_nan],
                                                                           axis=1)),
        'cpg': (['time', 'wavelength_number'], np.concatenate([binned.cpg, fill_nan], axis=1)),
        'cpg_ts': (['time', 'wavelength_number'], np.concatenate([binned.cpg_ts, fill_nan], axis=1)),
    }, coords={'time': (['time'], binned.time.values), 'wavelength_number': wavelength_number})

    # drop the original 2D variables from the binned data set
    drop = binned.drop_vars(['wavelength_number', 'wavelength_a', 'a_signal', 'a_reference',
                        'optical_absorption', 'apg', 'apg_ts', 'apg_ts_s',
                        'wavelength_c', 'c_signal', 'c_reference',
                        'beam_attenuation', 'cpg', 'cpg_ts'])

    # reset the data type for the 'a' and 'c' signal and reference dark values, and the other raw parameters
    int_arrays = ['a_signal_dark', 'a_reference_dark', 'c_signal_dark', 'c_reference_dark',
                  'internal_temp_raw', 'external_temp_raw', 'deployment', 'profile']
    for k in drop.variables:
        if k in int_arrays:
            drop[k] = drop[k].astype(int)

    # recombine the two datasets
    optaa = xr.merge([drop, ac])

    # reset the attributes, which the merging drops
    optaa.attrs = binned.attrs
    for v in optaa.variables:
        optaa[v].attrs = binned[v].attrs

    # reset some attributes
    for key, value in ATTRS.items():
        for atk, atv in value.items():
            if key in optaa.variables:
                optaa[key].attrs[atk] = atv

    # add the original variable name as an attribute, if renamed
    for key, value in rename.items():
        optaa[value].attrs['ooinet_variable_name'] = key

    # add the actual number of wavelengths to the dataset as an attribute
    optaa['wavelength_number'].attrs['actual_wavelengths'] = num_wavelengths

    # if the filter index was used to adjust the spectral jumps, add that attribute to the data set
    if cal.coeffs['grate_index']:
        optaa['a_jump_offsets'].attrs['grate_index'] = cal.coeffs['grate_index']
        optaa['c_jump_offsets'].attrs['grate_index'] = cal.coeffs['grate_index']

    return optaa


def main(argv=None):
    """
    Command line interface for processing OOI OPTAA NetCDF file(s) from the
    Endurance, Pioneer or Global surface moorings, or the Endurance surface
    piercing profilers. Creates a cleaned and processed xarray dataset of the
    OPTAA data saved to a NetCDF file.
    """
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    method = args.method
    stream = args.stream
    deploy = args.deploy
    start = args.start
    stop = args.stop

    # check if we are specifying a deployment or a specific date and time range
    if not deploy or (start and stop):
        return SyntaxError('You must specify either a deployment number or beginning and end dates of interest.')

    # if we are specifying a deployment number, then get the data from the Gold Copy THREDDS server
    if deploy:
        optaa = load_gc_thredds(site, node, sensor, method, stream, ('.*deployment%04d.*OPTAA.*\\.nc$' % deploy))
        cal_file = ('{}-{}-{}.deploy{:02d}.cal_coeffs.json'.format(site, node, sensor, deploy))

        # check to see if we downloaded any data
        if not optaa:
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

        # OPTAA data is different from other instruments. it needs to be processed on a per-deployment basis in order
        # to get the correct number of wavelengths before it can be merged into a single dataset. create a list of
        # all the files that were returned by the M2M request, and determine the deployments that are included in the
        # request
        files = list_files(r['allURLs'][0], '.+OPTAA.+\\.nc$')
        if not files:
            exit_text = ('Data unavailable for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                    stream, start, stop))
            raise SystemExit(exit_text)

        deployments = np.unique([int(sub.split('/')[3][10:14]) for sub in files])

        # loop through the deployments and download the data for each one
        optaa = []
        cal_file = []
        for deploy in deployments:
            # Valid M2M request, download the data on a per-deployment basis
            data = m2m_collect(r, ('.*deployment%04d.*OPTAA.*\\.nc$' % deploy))
            if data:
                optaa.append(data)
                cal_file.append('{}-{}-{}.deploy{:02d}.cal_coeffs.json'.format(site, node, sensor, deploy))

        # check to see if we downloaded any data (remove empty/none entries from the list)
        if not optaa:
            exit_text = ('Data unavailable for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                    stream, start, stop))
            raise SystemExit(exit_text)

    # set up the calibration file path
    out_file = os.path.abspath(args.outfile)
    cal_path = os.path.dirname(out_file)
    if not os.path.exists(cal_path):
        os.makedirs(cal_path)

    # clean-up and reorganize the data
    multi = isinstance(optaa, list)
    if node == 'SP001':
        # this OPTAA is part of a CSPP
        if multi:
            for i, ds in enumerate(optaa):
                cfile = os.path.join(cal_path, cal_file[i])
                optaa[i] = optaa_cspp(ds, cfile)
            optaa = xr.concat(optaa, dim='time')
        else:
            cal_file = os.path.join(cal_path, cal_file)
            optaa = optaa_cspp(optaa, cal_file)
    else:
        # this OPTAA is stand-alone on one of the moorings
        if multi:
            for i, ds in enumerate(optaa):
                cfile = os.path.join(cal_path, cal_file[i])
                optaa[i] = optaa_datalogger(ds, cfile)
            optaa = xr.concat(optaa, dim='time')
        else:
            cal_file = os.path.join(cal_path, cal_file)
            optaa = optaa_datalogger(optaa, cal_file)

    # get the vocabulary information for the site, node, and sensor and update the dataset attributes
    vocab = get_vocabulary(site, node, sensor)[0]
    optaa = optaa.sortby(['deployment', 'time'])
    optaa = update_dataset(optaa, vocab['maxdepth'])

    # save the data to disk
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    optaa.to_netcdf(out_file, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)


if __name__ == '__main__':
    main()
