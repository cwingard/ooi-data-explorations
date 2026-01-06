#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Updated metadata attributes for the PCO2A sensor after being merged and
    re-processed with QC checks applied.
"""
import numpy as np

PCO2A = {
    'global': {
        'title': 'Partial Pressure of CO2 in the Air and Surface Water from the OOI Endurance Array',
        'source': ('Surface observations of atmospheric and sea surface pCO2 along with sea surface temperature, '
                   'sea surface salinity, and 10-meter normalized wind speeds.'),
        'comment': ('Measurements of the partial pressure of carbon dioxide (pCO2) in the air and surface water '
                    'collected by the OOI Endurance Array using the Pro-Oceanus CO2-Pro ATM sensor combined with '
                    'data from the co-located ASIMET meteorolgical sensor suite measuring (among other variables) the '
                    'sea surface temperature, salinity and wind speed. Measurements have been re-processed from '
                    'the source data available on the OOI Gold Copy THREDDS catalog with automated and HITL quality '
                    'control flags applied to remove bad data as part of the processing.'),
        'references': ('https://oceanobservatories.org, https://pro-oceanus.com/products/pro-series/co2-pro-atm, '
                       'https://www.whoi.edu/what-we-do/explore/instruments/instruments-sensors-samplers/air-sea-interaction-meteorology-the-asimet-system/'),
    },
    'co2_mole_fraction_atm': {
        'long_name': 'Mole Fraction of Atmospheric CO2',
        'standard_name': 'mole_fraction_of_carbon_dioxide_in_dry_air',
        'units': 'ppm',
        'comment': ('The measured CO2 mole fraction (XCO2) in air or seawater internally computed onboard the sensor '
                    'from the raw measurements.'),
        'ooi_data_product_identifier': 'XCO2ATM_L0'
    },
    'gas_stream_pressure_atm': {
        'long_name': 'Total Gas Stream Pressure (Atmospheric)',
        'units': 'mbar',
        'comment': ('The gas stream pressure is the pressure of the internal gas volume of the pCO2 Air-Sea '
                   'instrument. This data product is used to calculate Partial Pressure of CO2 in air and seawater. '
                   'The gas stream pressure is measured for both the air and water measurements.')

    },
    'partial_pressure_co2_atm': {
        'long_name': 'Partial Pressure of CO2 in Air',
        'standard_name': 'surface_partial_pressure_of_carbon_dioxide_in_air',
        'units': 'uatm',
        'comment': ('The partial pressure of CO2 in air refers to the pressure that would be exerted by '
                    'CO2 if all other atmoshperhic gases were removed. The Partial pressure of CO2 (uatm) in air is '
                    'calculated from the CO2 mole fraction (ppm), the gas stream pressure (mbar) and the standard '
                    'atmospheric pressure set to a default of 1013.25 mbar/atm.'),
        'ooi_data_product_identifier': 'PCO2ATM_L1',
        'ancillary_variables': 'co2_mole_fraction_atm, gas_stream_pressure_atm'        
    },
    'co2_mole_fraction_ssw': {
        'long_name': 'Mole Fraction of Surface Seawater CO2',
        'standard_name': 'mole_fraction_of_carbon_dioxide_in_dry_air',
        'units': 'ppm',
        'comment': ('The measured CO2 mole fraction (XCO2) in air or seawater internally computed onboard the sensor '
                    'from the raw measurements.'),
        'ooi_data_product_identifier': 'XCO2SSW_L0'
    },
    'gas_stream_pressure_ssw': {
        'long_name': 'Total Gas Stream Pressure (Surface Water)',
        'units': 'mbar',
        'comment': ('The gas stream pressure is the pressure of the internal gas volume of the pCO2 Air-Sea '
                   'instrument. This data product is used to calculate Partial Pressure of CO2 in air and seawater.')
    },
    'partial_pressure_co2_ssw': {
        'long_name': 'Partial Pressure of CO2 in Seawater',
        'standard_name': 'surface_partial_pressure_of_carbon_dioxide_in_sea_water',
        'units': 'uatm',
        'comment': ('The partial pressure of a dissolved gas in sea water is the partial pressure in air with which '
                    'it would be in equilibrium. The partial pressure of a gaseous constituent of air is the pressure '
                    'that it would exert if all other gaseous constituents were removed, assuming the volume, the '
                    'temperature, and its number of moles remain unchanged.'),
        'ooi_data_product_identifier': 'PCO2SSW_L1',
        'ancillary_variables': 'mole_fraction_ssw, gas_stream_pressure_ssw'
    },
    'sea_surface_temperature': {
        'long_name': 'Sea Surface Temperature',
        'standard_name': 'sea_surface_temperature',
        'units': 'degrees_Celsius',
        'comment': ('Sea surface temperature is the in-situ temperature of the seawater near the ocean surface. This value '
                    'was recorded by the ASIMET bulk meteorology system and added to the pCO2 data, replacing the earler '
                    'values produced by the OOI system'),
        'ooi_data_product_identifier': 'TEMPSRF_L1'
    },
    'sea_surface_salinity': {
        'long_name': 'Sea Surface Practical Salinity',
        'standard_name': 'sea_surface_salinity',
        'units': '1',
        'comment': ('Salinity is generally defined as the concentration of dissolved salt in a parcel of sea water. '
                    'Practical Salinity is a more specific unitless quantity calculated from the conductivity of '
                    'sea water and adjusted for temperature and pressure. It is approximately equivalent to Absolute '
                    'Salinity (the mass fraction of dissolved salt in sea water), but they are not interchangeable. This '
                    'value was recorded by the ASIMET bulk meteorology system and added to the pCO2 data, replacing the '
                    'earler values produced by the OOI system'),
        'ooi_data_product_identifier': 'SALSURF_L2'
    },
    'normalized_10m_wind': {
        'long_name': 'Normalized 10-meter Wind Speed',
        'standard_name': 'wind_speed',
        'units': 'm s-1',
        'comment': ('The 10-meter wind speed is the wind speed measured at a height of 10 meters above the sea '
                    'surface. The 10-meter wind speed is calculated from the 2D wind vector using the method '
                    'described in the WMO Guide to Meteorological Instruments and Methods of Observation, '
                    'Chapter 7, section 7.2.1. The 10-meter wind speed is used to calculate the gas transfer '
                    'velocity of CO2 across the air-sea interface. This value was recorded by the ASIMET bulk '
                    'meteorology system and added to the pCO2 data, replacing the earler values produced by the '
                    'OOI system'),
        'ooi_data_product_identifier': 'WIND10M_L2'
    },
    'sea_to_air_co2_flux': {
        'long_name': 'Sea Surface to Atmosphere CO2 Flux',
        'standard_name': 'surface_upward_mole_flux_of_carbon_dioxide',
        'units': 'umol m-2 s-1',
        'comment': ('Flux of CO2 across the air-sea interface. The CO2 flux is calculated from the partial pressure '
                    'of CO2 in air and seawater, the 10-meter wind speed, the sea surface temperature, and the sea'
                    'surface salinity using the method described in Wanninkhof (1992). Positive values indicate a '
                    'flux from the ocean to the atmosphere. Negative values indicate a flux from the atmosphere to '
                    'the ocean. This value replaces the one produced by the OOI system after applying QC flags and '
                    'recalculating with the cleaned surface temperature, salinity and normalized 10-meter wind data.'),
        'ooi_data_product_identifier': 'CO2FLUX_L2',
        'ancillary_variables': 'partial_pressure_co2_atm, partial_pressure_co2_ssw, sea_surface_temperature, sea_surface_salinity, normalized_10m_wind'
    },
    'alkalinity': {
        'long_name': 'Estimated Total Alkalinity',
        'standard_name': 'sea_water_alkalinity_per_unit_mass_expressed_as_mole_equivalent',
        'units': 'umol kg-1',
        'comment': ('The standard name sea_water_alkalinity_per_unit_mass_expressed_as_mole_equivalent is the total alkalinity '
                    'equivalent concentration (including carbonate, nitrogen, silicate, and borate components) expressed as the '
                    'number of moles of alkalinity per unit mass of seawater. This value is an estimate based on the total '
                    'alkalinity to salinity developed by Fassbender et al. (2017).'),
        'references': 'https://doi.org/10.1007/s12237-016-0168-z',
        'ancillary_variables': 'sea_surface_salinity'
    },
    'seawater_ph': {
        'long_name': 'Estimated Seawater pH on the Total Scale',
        'standard_name': 'sea_water_ph_reported_on_total_scale',
        'units': '1',
        'comment': ('Measure of the acidity of seawater, defined as the negative logarithm of the concentration of dissolved hydrogen '
                    'ions plus bisulfate ions in a sea water medium; it can be measured or calculated; when measured the scale is '
                    'defined according to a series of buffers prepared in artificial seawater containing bisulfate. The quantity may '
                    'be written as pH(total) = -log([H+](free) + [HSO4-]). This value has been calculated using the estimated total '
                    'alkalinity and measured sea water pCO2 using pyCO2SYS.'),
        'references': 'https://doi.org/10.5194/gmd-15-15-2022',
        'ancillary_variables': 'partial_pressure_co2_ssw, alkalinity, sea_surface_salinity, sea_surface_temperature'
    },
    'aragonite_saturation': {
        'long_name': 'Estimated Aragonite Saturation State',
        'units': '1',
        'comment': ('Aragonite saturation is a measure of how much aragonite (a form of calcium carbonate) is available in seawater '
                    'for marine organisms like corals, shellfish, and pteropods to build their shells and skeletons; a high saturation '
                    'state (>1) means it is easier to build, while low saturation (<1, undersaturation) makes it harder and can cause '
                    'existing shells to dissolve, a key impact of ocean acidification. This value has been calculated using the estimated '
                    'total alkalinity and the measured sea surface pCO2 using PyCO2SYS.'),
        'references': 'https://doi.org/10.5194/gmd-15-15-2022',
        'ancillary_variables': 'partial_pressure_co2_ssw, alkalinity, sea_surface_salinity, sea_surface_temperature'
    }
}

ENCODINGS = {
    'co2_mole_fraction_atm': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'gas_stream_pressure_atm': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'partial_pressure_co2_atm': {
        'missing_value': np.nan,
        '_FillValue': np.nan        
    },
    'co2_mole_fraction_ssw': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'gas_stream_pressure_ssw': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'partial_pressure_co2_ssw': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_surface_temperature': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_surface_salinity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'normalized_10m_wind': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_to_air_co2_flux': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'alkalinity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'seawater_ph': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'aragonite_saturation': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    }    
}#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Updated metadata attributes for the PCO2A sensor after being merged and
    re-processed with QC checks applied.
"""
import numpy as np

PCO2A = {
    'global': {
        'title': 'Partial Pressure of CO2 in the Air and Surface Water from the OOI Endurance Array',
        'project': 'U.S. National Science Foundation (NSF) Ocean Observatories Initiative (OOI)',
        'institution': 'OOI Endurance Array (EA)',
        'source': ('Surface observations of atmospheric and sea surface pCO2 along with sea surface temperature, '
                   'sea surface salinity, and 10-meter normalized wind speeds.'),
        'comment': ('Measurements of the partial pressure of carbon dioxide (pCO2) in the air and surface water '
                    'collected by the OOI Endurance Array using the Pro-Oceanus CO2-Pro ATM sensor combined with '
                    'data from the co-located ASIMET meteorolgical sensor suite measuring (among other variables) the '
                    'sea surface temperature, salinity and wind speed. Measurements have been re-processed from '
                    'the source data available on the OOI Gold Copy THREDDS catalog with automated and HITL quality '
                    'control flags applied to remove bad data.'),
        'references': ('https://oceanobservatories.org, https://pro-oceanus.com/products/pro-series/co2-pro-atm, '
                       'https://www.whoi.edu/what-we-do/explore/instruments/instruments-sensors-samplers/air-sea-interaction-meteorology-the-asimet-system/'),
        'acknowledgement': 'U.S. National Science Foundation (NSF)',
        'creator_name': 'U.S. National Science Foundation (NSF) Ocean Observatories Initiative (OOI)',
        'creator_email': 'helpdesk@oceanobservatories.org',
        'creator_url': 'https://oceanobservatories.org',
        'featureType': 'timeSeries',
        'cdm_data_type': 'Station',
        'Conventions': 'CF-1.7'
    },
    'co2_mole_fraction_atm': {
        'long_name': 'Mole Fraction of Atmospheric CO2',
        'standard_name': 'mole_fraction_of_carbon_dioxide_in_dry_air',
        'units': 'ppm',
        'comment': ('The measured CO2 mole fraction (XCO2) in air or seawater internally computed onboard the sensor '
                    'from the raw measurements.'),
        'ooi_data_product_identifier': 'XCO2ATM_L0'
    },
    'gas_stream_pressure_atm': {
        'long_name': 'Total Gas Stream Pressure (Atmospheric)',
        'units': 'mbar',
        'comment': ('The gas stream pressure is the pressure of the internal gas volume of the pCO2 Air-Sea '
                   'instrument. This data product is used to calculate Partial Pressure of CO2 in air and seawater. '
                   'The gas stream pressure is measured for both the air and water measurements.')

    },
    'partial_pressure_co2_atm': {
        'long_name': 'Partial Pressure of CO2 in Air',
        'standard_name': 'surface_partial_pressure_of_carbon_dioxide_in_air',
        'units': 'uatm',
        'comment': ('The partial pressure of CO2 in air refers to the pressure that would be exerted by '
                    'CO2 if all other atmoshperhic gases were removed. The Partial pressure of CO2 (uatm) in air is '
                    'calculated from the CO2 mole fraction (ppm), the gas stream pressure (mbar) and the standard '
                    'atmospheric pressure set to a default of 1013.25 mbar/atm.'),
        'ooi_data_product_identifier': 'PCO2ATM_L1',
        'ancillary_variables': 'co2_mole_fraction_atm, gas_stream_pressure_atm'        
    },
    'co2_mole_fraction_ssw': {
        'long_name': 'Mole Fraction of Surface Seawater CO2',
        'standard_name': 'mole_fraction_of_carbon_dioxide_in_dry_air',
        'units': 'ppm',
        'comment': ('The measured CO2 mole fraction (XCO2) in air or seawater internally computed onboard the sensor '
                    'from the raw measurements.'),
        'ooi_data_product_identifier': 'XCO2SSW_L0'
    },
    'gas_stream_pressure_ssw': {
        'long_name': 'Total Gas Stream Pressure (Surface Water)',
        'units': 'mbar',
        'comment': ('The gas stream pressure is the pressure of the internal gas volume of the pCO2 Air-Sea '
                   'instrument. This data product is used to calculate Partial Pressure of CO2 in air and seawater.')
    },
    'partial_pressure_co2_ssw': {
        'long_name': 'Partial Pressure of CO2 in Seawater',
        'standard_name': 'surface_partial_pressure_of_carbon_dioxide_in_sea_water',
        'units': 'uatm',
        'comment': ('The partial pressure of a dissolved gas in sea water is the partial pressure in air with which '
                    'it would be in equilibrium. The partial pressure of a gaseous constituent of air is the pressure '
                    'that it would exert if all other gaseous constituents were removed, assuming the volume, the '
                    'temperature, and its number of moles remain unchanged.'),
        'ooi_data_product_identifier': 'PCO2SSW_L1',
        'ancillary_variables': 'mole_fraction_ssw, gas_stream_pressure_ssw'
    },
    'sea_surface_temperature': {
        'long_name': 'Sea Surface Temperature',
        'standard_name': 'sea_surface_temperature',
        'units': 'degrees_Celsius',
        'comment': ('Sea surface temperature is the in-situ temperature of the seawater near the ocean surface. This value '
                    'was recorded by the ASIMET bulk meteorology system and added to the pCO2 data, replacing the earler '
                    'values produced by the OOI system'),
        'ooi_data_product_identifier': 'TEMPSRF_L1'
    },
    'sea_surface_salinity': {
        'long_name': 'Sea Surface Practical Salinity',
        'standard_name': 'sea_surface_salinity',
        'units': '1',
        'comment': ('Salinity is generally defined as the concentration of dissolved salt in a parcel of sea water. '
                    'Practical Salinity is a more specific unitless quantity calculated from the conductivity of '
                    'sea water and adjusted for temperature and pressure. It is approximately equivalent to Absolute '
                    'Salinity (the mass fraction of dissolved salt in sea water), but they are not interchangeable. This '
                    'value was recorded by the ASIMET bulk meteorology system and added to the pCO2 data, replacing the '
                    'earler values produced by the OOI system'),
        'ooi_data_product_identifier': 'SALSURF_L2'
    },
    'normalized_10m_wind': {
        'long_name': 'Normalized 10-meter Wind Speed',
        'standard_name': 'wind_speed',
        'units': 'm s-1',
        'comment': ('The 10-meter wind speed is the wind speed measured at a height of 10 meters above the sea '
                    'surface. The 10-meter wind speed is calculated from the 2D wind vector using the method '
                    'described in the WMO Guide to Meteorological Instruments and Methods of Observation, '
                    'Chapter 7, section 7.2.1. The 10-meter wind speed is used to calculate the gas transfer '
                    'velocity of CO2 across the air-sea interface. This value was recorded by the ASIMET bulk '
                    'meteorology system and added to the pCO2 data, replacing the earler values produced by the '
                    'OOI system'),
        'ooi_data_product_identifier': 'WIND10M_L2'
    },
    'sea_to_air_co2_flux': {
        'long_name': 'Sea Surface to Atmosphere CO2 Flux',
        'standard_name': 'surface_upward_mole_flux_of_carbon_dioxide',
        'units': 'umol m-2 s-1',
        'comment': ('Flux of CO2 across the air-sea interface. The CO2 flux is calculated from the partial pressure '
                    'of CO2 in air and seawater, the 10-meter wind speed, the sea surface temperature, and the sea'
                    'surface salinity using the method described in Wanninkhof (1992). Positive values indicate a '
                    'flux from the ocean to the atmosphere. Negative values indicate a flux from the atmosphere to '
                    'the ocean. This value replaces the one produced by the OOI system after applying QC flags and '
                    'recalculating with the cleaned surface temperature, salinity and normalized 10-meter wind data.'),
        'ooi_data_product_identifier': 'CO2FLUX_L2',
        'ancillary_variables': 'partial_pressure_co2_atm, partial_pressure_co2_ssw, sea_surface_temperature, sea_surface_salinity, normalized_10m_wind'
    },
    'alkalinity': {
        'long_name': 'Estimated Total Alkalinity',
        'standard_name': 'sea_water_alkalinity_per_unit_mass_expressed_as_mole_equivalent',
        'units': 'umol kg-1',
        'comment': ('The standard name sea_water_alkalinity_per_unit_mass_expressed_as_mole_equivalent is the total alkalinity '
                    'equivalent concentration (including carbonate, nitrogen, silicate, and borate components) expressed as the '
                    'number of moles of alkalinity per unit mass of seawater. This value is an estimate based on the total '
                    'alkalinity to salinity developed by Fassbender et al. (2017).'),
        'references': 'https://doi.org/10.1007/s12237-016-0168-z',
        'ancillary_variables': 'sea_surface_salinity'
    },
    'seawater_ph': {
        'long_name': 'Estimated Seawater pH on the Total Scale',
        'standard_name': 'sea_water_ph_reported_on_total_scale',
        'units': '1',
        'comment': ('Measure of the acidity of seawater, defined as the negative logarithm of the concentration of dissolved hydrogen '
                    'ions plus bisulfate ions in a sea water medium; it can be measured or calculated; when measured the scale is '
                    'defined according to a series of buffers prepared in artificial seawater containing bisulfate. The quantity may '
                    'be written as pH(total) = -log([H+](free) + [HSO4-]). This value has been calculated using the estimated total '
                    'alkalinity and measured sea water pCO2 using pyCO2SYS.'),
        'references': 'https://doi.org/10.5194/gmd-15-15-2022',
        'ancillary_variables': 'partial_pressure_co2_ssw, alkalinity, sea_surface_salinity, sea_surface_temperature'
    },
    'aragonite_saturation': {
        'long_name': 'Estimated Aragonite Saturation State',
        'units': '1',
        'comment': ('Aragonite saturation is a measure of how much aragonite (a form of calcium carbonate) is available in seawater '
                    'for marine organisms like corals, shellfish, and pteropods to build their shells and skeletons; a high saturation '
                    'state (>1) means it is easier to build, while low saturation (<1, undersaturation) makes it harder and can cause '
                    'existing shells to dissolve, a key impact of ocean acidification. This value has been calculated using the estimated '
                    'total alkalinity and the measured sea surface pCO2 using PyCO2SYS.'),
        'references': 'https://doi.org/10.5194/gmd-15-15-2022',
        'ancillary_variables': 'partial_pressure_co2_ssw, alkalinity, sea_surface_salinity, sea_surface_temperature'
    }
}

ENCODINGS = {
    'co2_mole_fraction_atm': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'gas_stream_pressure_atm': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'partial_pressure_co2_atm': {
        'missing_value': np.nan,
        '_FillValue': np.nan        
    },
    'co2_mole_fraction_ssw': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'gas_stream_pressure_ssw': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'partial_pressure_co2_ssw': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_surface_temperature': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_surface_salinity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'normalized_10m_wind': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'sea_to_air_co2_flux': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'alkalinity': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'seawater_ph': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    },
    'aragonite_saturation': {
        'missing_value': np.nan,
        '_FillValue': np.nan
    }    
}