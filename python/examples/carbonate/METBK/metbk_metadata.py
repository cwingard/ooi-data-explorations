#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Updated metadata attributes for the METBK dataset after being merged
    and re-processed with QC checks applied.
"""
import numpy as np

METBK = {
    'global': {
        'title': 'Bulk Meteorological (METBK) Measurements from the OOI Endurance Array',
        'source': ('Surface observations of meteorological sensors'),
        'comment': ('Measurements of the surface meteorology sensors collected by the OOI Endurance Array at a 1-minute '
                    'sample rate using the ASIMET meteorolgical sensor suite produced by the UOP group at WHOI. '
                    'Measures surface meteorology and provides the data required to compute air-sea fluxes of heat, '
                    'freshwater, and momentum. Measurements have been re-processed from the source data available on '
                    'the OOI Gold Copy THREDDS catalog with automated and HITL quality control flags applied to remove '
                    'bad data. Additionally, recovered instrument data from the CT sensor (SBE37 MicroCAT) is used to '
                    'replace missing CT data in the data logger data sets, where available.'),
        'references': ('https://oceanobservatories.org, https://uop.whoi.edu/, '
                       'https://www.whoi.edu/what-we-do/explore/instruments/instruments-sensors-samplers/air-sea-interaction-meteorology-the-asimet-system/')
    },
    'barometric_pressure': {
        'long_name': 'Barometric Pressure',
        'standard_name': 'air_pressure',
        'units': 'mbar',
        'comment': ('Barometric Pressure is a measure of the weight of the column of air above the sensor. It is '
                    'also commonly referred to as atmospheric pressure.'),
        'ooi_data_product_identifier': 'BARPRES_L1',
    },
    'relative_humidity': {
        'long_name': 'Relative Humidity',
        'standard_name': 'relative_humidity',
        'units': 'percent',
        'comment': ('Relative humidity is the ratio of the current absolute humidity to the highest possible '
                    'absolute humidity, which depends on the current air temperature.'),
        'ooi_data_product_identifier': 'RELHUMI_L1',
    },
    'air_temperature': {
        'long_name': 'Air Temperature',
        'standard_name': 'air_temperature',
        'units': 'degrees_Celsius',
        'comment': ('Air temperature refers to the temperature of the air surrounding the sensor; this is also '
                    'referred to as bulk temperature.'),
        'ooi_data_product_identifier': 'TEMPAIR_L1',
    },
    'precipitation': {
        'long_name': 'Hourly Precipitation Rate',
        'standard_name': 'lwe_precipitation_rate',
        'units': 'mm hr-1',
        'comment': ('Siphoning rain gauge measurements. Values cycle from 0 to 50 mm as the water level rises and '
                    'then is siphoned off. In converting to the rain rate, only positive increases greater than 0.25 mm '
                    'are used.'),
        'ooi_data_product_identifier': 'RAINRTE_L1',
    },
    'sea_surface_temperature': {
        'long_name': 'Sea Surface Temperature',
        'standard_name': 'sea_surface_temperature',
        'units': 'degrees_Celsius',
        'comment': 'Sea surface temperature is the in-situ temperature of the seawater near the ocean surface.',
        'ooi_data_product_identifier': 'TEMPSRF_L1',
    },
    'sea_surface_conductivity': {
        'long_name': 'Sea Surface Conductivity',
        'standard_name': 'sea_water_electrical_conductivity',
        'units': 'S m-1',
        'comment': ('Sea surface conductivity refers to the ability of seawater to conduct electricity. The presence '
                    'of ions, such as salt, increases the electrical conducting ability of seawater. As such, '
                    'conductivity can be used as a proxy for determining the quantity of salt in a sample of '
                    'seawater measured near the sea surface.'),
        'ooi_data_product_identifier': 'CONDSRF_L1',
    },
    'sea_surface_salinity': {
        'long_name': 'Sea Surface Practical Salinity',
        'standard_name': 'sea_surface_salinity',
        'units': '1',
        'comment': ('Salinity is generally defined as the concentration of dissolved salt in a parcel of sea water. '
                    'Practical Salinity is a more specific unitless quantity calculated from the conductivity of '
                    'sea water and adjusted for temperature and pressure. It is approximately equivalent to Absolute '
                    'Salinity (the mass fraction of dissolved salt in sea water), but they are not interchangeable.'),
        'ooi_data_product_identifier': 'SALSURF_L2',
        'ancillary_variables': 'sea_surface_conductivity, sea_surface_temperature',
    },
    'shortwave_irradiance': {
        'long_name': 'Downwelling Shortwave Irradiance',
        'standard_name': 'downwelling_shortwave_flux_in_air',
        'units': 'W m-2',
        'comment': ('Downwelling short-wave radiation at the surface has a component due to the direct solar beam, '
                    'and a diffuse component scattered from atmospheric constituents and reflected from clouds.'),
        'ooi_data_product_identifier': 'SHRTIRR_L1',
    },
    'longwave_irradiance': {
        'long_name': 'Downwelling Longwave Irradiance',
        'standard_name': 'downwelling_longwave_flux_in_air',
        'units': 'W m-2',
        'comment': ('Downwelling longwave radiation flux at the surface. Significant sources of longwave radiation in '
                    'hydrologic applications include the atmosphere itself, and any clouds that may be present '
                    'locally in the atmosphere. Clouds usually have a higher heat content and higher temperature '
                    'than clear atmosphere, and therefore there is increased downwelling longwave radiation on '
                    'cloudy days'),
        'ooi_data_product_identifier': 'LONGIRR_L1',
    },
    'eastward_wind_velocity': {
        'long_name': 'Eastward Wind Velocity',
        'standard_name': 'eastward_wind',
        'units': 'm s-1',
        'comment': 'Eastward wind velocity corrected for magnetic declination and the identified underspeeding issue.',
        'ooi_data_product_identifier': 'WINDAVG-VLE_L1',
    },
    'northward_wind_velocity': {
        'long_name': 'Northward Wind Velocity',
        'standard_name': 'northward_wind',
        'units': 'm s-1',
        'comment': 'Northward wind velocitycorrected for magnetic declination and the identified underspeeding issue.',
        'ooi_data_product_identifier': '"WINDAVG-VLN_L1',
    }
}

ENCODINGS = {
    'barometric_pressure': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'relative_humidity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'air_temperature': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'precipitation': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_surface_temperature': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_surface_conductivity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_surface_salinity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'shortwave_irradiance': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'longwave_irradiance': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'eastward_wind_velocity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'northward_wind_velocity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    }
}