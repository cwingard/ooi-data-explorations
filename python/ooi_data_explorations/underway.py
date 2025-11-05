#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Andrew Reed
@brief Provides methods for parsing ship underway data
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Union

# --- Dataset Attributes ---

ATTRS = {
    "latitude": {
        "long_name": 'Latitude',
        'standard_name': 'latitude',
        'units': 'degrees' },
    'longitude': {
        'long_name': 'Longitude',
        'standard_name': 'longitude',
        'units': 'degrees' },
    'ship_speed': {
        'long_name': 'Ship Speed',
        'units': 'kts' },
    'heading' : {
        'long_name': 'Heading',
        'units': 'degrees' },
    'course_over_ground': {
        'long_name': 'Course over Ground',
        'units': 'degrees',
        'comment': 'Ship course over ground from GPS.' },
    'speed_over_ground': {
        'long_name': 'Speed over Ground',
        'units': 'kts',
        'comment': 'Ship speed over ground from GPS.' },
    'air_temperature_port': {
        'standard_name': 'air_temperature',
        'long_name': 'Bulk Air Temperature - Port',
        'units': 'degrees_Celcius',
        'comment': ('Air Temperature refers to the temperature of the air surrounding the sensor; this is also referred to as bulk temperature. '
                    'This measurement originates from the Port side of the ship met sensor array. It has not been adjusted for height.') },
    'air_temperature_starboard': {
        'standard_name': 'air_temperature',
        'long_name': 'Bulk Air Temperature - Starboard',
        'units': 'degrees_Celcius',
        'comment': ('Air Temperature refers to the temperature of the air surrounding the sensor; this is also referred to as bulk temperature. '
                    'This measurement originates from the starboard side of the ship met sensor array. It has not been adjusted for height.') },
    'air_pressure_port': {
        'standard_name': 'air_pressure',
        'long_name': 'Air Pressure - Port',
        'units': 'hPa',
        'comment': ('Air Pressure is a measure of the force per unit area of the column of air above the sensor. This measurement originates from '
                    'the port side of the ship met sensor array. It has not been adjusted for height.') },
    'air_pressure_starboard': {
        'standard_name': 'air_pressure',
        'long_name': 'Air Pressure - Starboard',
        'units': 'hPa',
        'comment': ('Air Pressure is a measure of the force per unit area of the column of air above the sensor. This measurement originates from '
                    'the starboard side of the ship met sensor array. It has not been adjusted for height.') },
    'precipitation_rate_port': {
        'standard_name': 'precipitation_rate',
        'long_name': 'Precipitation Rate - Port',
        'units': 'mm/hr',
        'comment': ('Precipitation rate is a measure of the thickness of a layer of water that accumulates per unit time. '
                    'This measurement originates from the port side of the ship met sensor array.' ) },
    'precipitation_rate_starboard': {
        'standard_name': 'precipitation_rate',
        'long_name': 'Precipitation Rate - Starboard',
        'units': 'mm/hr',
        'comment': ('Precipitation rate is a measure of the thickness of a layer of water that accumulates per unit time. '
                    'This measurement originates from the starboard side of the ship met sensor array.' ) },
    'precipitation_amount_port': {
        'standard_name': 'thickness_of_precipitation_amount',
        'long_name': 'Amount of Precipitation - Port',
        'units': 'mm',
        'comment' : ('Precipitation amount is a measure of the thickness of the total water that accumulates. '
                     'This measurement originates from the port side of the ship met sensor array.') },
    'precipitation_amount_starboard': {
        'standard_name': 'thickness_of_precipitation_amount',
        'long_name': 'Amount of Precipitation - Starboard',
        'units': 'mm',
        'comment' : ('Precipitation amount is a measure of the thickness of the total water that accumulates. '
                     'This measurement originates from the starboard side of the ship met sensor array.') },
    'relative_wind_direction_port': {
        'standard_name': 'wind_to_direction',
        'long_name': 'Relative Wind Direction - Port',
        'units': 'degrees',
        'comment': ('The relative direction of the wind is given as the direction to which the wind is blowing, '
                    'uncorrected for ship motion. This measurement originates from the port side of the ship '
                    'met sensor array.') },
    'relative_wind_direction_starboard': {
        'standard_name': 'wind_to_direction',
        'long_name': 'Relative Wind Direction - Starboard',
        'units': 'degrees',
        'comment': ('The relative direction of the wind is given as the direction to which the wind is blowing, '
                    'uncorrected for ship motion. This measurement originates from the starboard side of the ship '
                    'met sensor array.') },    
    'relative_wind_speed_port': {
        'standard_name': 'wind_speed',
        'long_name': 'Relative Wind Speed - Port',
        'units': 'm s-1',
        'comments': ('Relative wind speed is the magnitude of the velocity, uncorrected for ship motion. '
                     'This measurements originates from the port side of the ship met sensor array.') },
    'relative_wind_speed_starboard': {
        'standard_name': 'wind_speed',
        'long_name': 'Relative Wind Speed - Starboard',
        'units': 'm s-1',
        'comments': ('Relative wind speed is the magnitude of the velocity, uncorrected for ship motion. '
                     'This measurements originates from the starboard side of the ship met sensor array.') },
    'relative_humidity_port': {
        'standard_name': 'relative_humidity',
        'long_name': 'Relative Humidity - Port',
        'units': 'percent',
        'comments': ('This measurements originates from the port side of the ship met sensor array.') },
    'relative_humidity_starboard': {
        'standard_name': 'relative_humidity',
        'long_name': 'Relative Humidity - Starboard',
        'units': 'percent',
        'comments': ('This measurements originates from the starboard side of the ship met sensor array.') },
    'wind_speed_port': {
        'standard_name': 'wind_speed',
        'long_name': 'Wind Speed - Port',
        'units': 'm s-1',
        'comment': ('Wind speed is the magnitude of the velocity, corrected for ship motion. '
                    'This measurement originates from the port side of the ship met sensor array.') },
    'wind_speed_starboard': {
        'standard_name': 'wind_speed',
        'long_name': 'Wind Speed - Starboard',
        'units': 'm s-1',
        'comment': ('Wind speed is the magnitude of the velocity, corrected for ship motion. '
                    'This measurement originates from the starboard side of the ship met sensor array.') },   
    'wind_direction_port': {
        'standard_name': 'wind_to_direction',
        'long_name': ' Wind Direction - Port',
        'units': 'degrees',
        'comment':('The direction of the wind is given as the direction to which the wind is blowing, '
                    'corrected for ship motion and relative to magnetic north. This measurement originates '
                    'from the port side of the ship met sensor array.') },
    'wind_direction_starboard': {
        'standard_name': 'wind_to_direction',
        'long_name': 'Wind Direction - Starboard',
        'units': 'degrees',
        'comment': ('The direction of the wind is given as the direction to which the wind is blowing, '
                    'corrected for ship motion and relative to magnetic north. This measurement originates '
                    'from the starboard side of the ship met sensor array.') }, 
    'barometric_pressure_port': {
        'standard_name': 'air_pressure',
        'long_name': 'Barometric Pressure - Port',
        'units': 'hPa',
        'comment': ('Barometric Pressure is a measure of the weight of the column of air above the sensor. '
                    'It is also commonly referred to as atmospheric pressure. This measurement originates '
                    'from the port side of the ship met sensor array and is adjusted to the height of the '
                    'sensor array.') },
    'barometric_pressure_starboard': {
        'standard_name': 'air_pressure',
        'long_name': 'Barometric Pressure - Starboard',
        'units': 'hPa',
        'comment': ('Barometric Pressure is a measure of the weight of the column of air above the sensor. '
                    'It is also commonly referred to as atmospheric pressure. This measurement originates '
                    'from the starboard side of the ship met sensor array and is adjusted to the height of the '
                    'sensor array.') },
    'shortwave_radiation': {
        'standard_name': 'downwelling_shortwave_flux_in_air',
        'long_name': 'Downwelling Shortwave Flux',
        'units': 'W m-2',
        'comment': ('Downwelling radiation is radiation from above with positive sign downwards. It does not mean "net downward".') },
    'longwave_radiation': {
        'standard_name': 'downwelling_longwave_flux_in_air',
        'long_name': 'Downwelling Longwave Flux',
        'units': 'W m-2',
        'comment': ('Downwelling radiation is radiation from above with positive sign downwards. It does not mean "net downward".') },
    'par': {
        'long_name': 'Photosynthetically Active Radiation',
        'units': 'uE m-2 s-1',
        'comment': ('Photosynthetically Active Radiation (PAR) is light of wavelengths 400-700 nm and is the portion of the light '
                    'spectrum utilized for photosynthesis. It is measured as photon flux density, which is the rate that umol of quanta '
                    'of light land per unit area.') },
    'sea_surface_salinity': {
        'standard_name': 'sea_surface_salinity',
        'long_name': 'Sea Surface Practical Salinity',
        'units': 'psu',
        'comment' : ('Salinity is generally defined as the concentration of dissolved salt in a parcel of seawater. Practical Salinity is '
                     'a more specific unitless quantity calculated from the conductivity of seawater and adjusted for temperature and pressure. '
                     'It is approximately equivalent to Absolute Salinity (the mass fraction of dissolved salt in seawater) but they are not '
                     'interchangeable. This measurement is made at 5m below the ship water line.') },
    'sea_surface_temperature': {
        'standard_name': 'sea_surface_temperature',
        'long_name': 'Sea Surface Temperature',
        'units': 'degrees_Celcius',
        'comment': ('Sea Surface Temperature is the temperature of the seawater near the ocean surface. This measurement is made at 5m below the '
                    'ship water line.') },
    'speed_of_sound': {
        'standard_name': 'speed_of_sound_in_seawater',
        'long_name': 'Speed of Sound in Water',
        'units': 'm s-1',
        'comment': ('This is the magnitude of the velocity of sound in the sea surface water.') },
    'depth12': {
        'long_name': 'Bottom Depth - 12kHz',
        'units': 'm',
        'comment': ('This is the calculated bottom depth from the 12kHz acoustic sensor.') },
    'depth35': {
        'long_name': 'Bottom Depth - 3.5kHz',
        'units': 'm',
        'comment': ('This is the calculated bottom depth from the 3.5kHz acoustic sensor.') },
    'em122': {
        'long_name': 'Bottom Depth - 12kHz multibeam center',
        'units': 'm',
        'comment': ('This is the calculated bottom depth from the 12kHz multibeam center return.') }
}

# --- Name mapping of parameters ---
name_map = {
    'Dec_LAT': 'latitude',
    'Dec_LON': 'longitude',
    'SPD': 'ship_speed',
    'HDT': 'heading',
    'COG': 'course_over_ground',
    'SOG': 'speed_over_ground',
    'WXTP_Ta': 'air_temperature_port',
    'WXTS_Ta': 'air_temperature_starboard',
    'WXTP_Pa': 'air_pressure_port',
    'WXTS_Pa': 'air_pressure_starboard',
    'WXTP_Ri': 'precipitation_rate_port',
    'WXTS_Ri': 'precipitation_rate_starboard',
    'WXTP_Rc': 'precipitation_amount_port',
    'WXTS_Rc': 'precipitation_amount_starboard',
    'WXTP_Dm': 'relative_wind_direction_port',
    'WXTS_Dm': 'relative_wind_direction_starboard',
    'WXTP_Sm': 'relative_wind_speed_port',
    'WXTS_Sm': 'relative_wind_speed_starboard',
    'WXTP_Ua': 'relative_humidity_port',
    'WXTS_Ua': 'relative_humidity_starboard',
    'WXTP_TS': 'wind_speed_port',
    'WXTS_TS': 'wind_speed_starboard',
    'WXTP_TD': 'wind_direction_port',
    'WXTS_TD': 'wind_direction_starboard',
    'BAROM_P': 'barometric_pressure_port',
    'BAROM_S': 'barometric_pressure_starboard',
    'RAD_SW': 'shortwave_radiation',
    'RAD_LW': 'longwave_radiation',
    'PAR': 'par',
    'SBE45S': 'sea_surface_salinity',
    'SBE48T': 'sea_surface_temperature',
    'SSVdslog': 'speed_of_sound',
    'Depth12': 'depth12',
    'Depth35': 'depth35',
    'EM122': 'em122'
}

# --- Utility functions ---
def _parse_key_value(line: str) -> Optional[tuple[str, str]]:
    """Helper to parse 'key: value' lines and normalize keys."""
    if ':' not in line:
        return None
    key, value = line.split(':', 1)
    key = '_'.join(key.strip().lower().split())
    return key, value.strip()


def _update_attrs(attrs: Dict[str, dict], keys: List[str], key: str, value: str) -> None:
    """Update multiple attribute dictionaries with the same key/value."""
    for attr in keys:
        attrs[attr].update({key: value})


# --- Main parsing functions ---
def parse_data_files(file_list: List[str]) -> pd.DataFrame:
    """
    Parse underway `.csv` files into a cleaned and indexed DataFrame.

    Parameters
    ----------
    file_list : list of str
        Paths to underway CSV data files. Each file should have
        `DATE_GMT` and `TIME_GMT` columns along with ship
        meteorological and oceanographic variables.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by datetime with cleaned variable names.
        Unnecessary columns are dropped and placeholder values
        (' NAN', ' NODATA') are replaced with NaN.

    Notes
    -----
    - Combines multiple CSVs into a single DataFrame.
    - Creates a unified `time` column from `DATE_GMT` and `TIME_GMT`.
    - Drops `DATE_GMT`, `TIME_GMT`, `FLR`, and `FLOW` if present.
    - Sorts by time and sets the index to `time`.

    Examples
    --------
    >>> df = parse_data_files(["file1.csv", "file2.csv"])
    >>> df.head()
                     latitude  longitude  ship_speed
    time
    2024-01-01 00:00     ...        ...         ...
    """
    # Find and load csv files into dataframes
    dfs = []
    for file in file_list:
        if file.endswith(".csv"):
            dfs.append(pd.read_csv(file, header=1))
            
    # If no csv files, return empty dataset
    if not dfs:
        return pd.DataFrame()
    else:
        df = pd.concat(dfs, ignore_index=True)

    # Clean columns
    df.columns = df.columns.str.strip()
        
    # Combine date and time
    df["TIME_GMT"] = df["TIME_GMT"].str.replace(":60.000", ":59.999", regex=False)
    df["time"] = pd.to_datetime(df["DATE_GMT"] + df["TIME_GMT"])
    
    # Sort values and reset index
    df = df.sort_values("time").set_index("time", drop=True)

    # Drop unwanted parameters if present
    df = df.drop(columns=[c for c in ["DATE_GMT", "TIME_GMT", "FLR", "FLOW"] if c in df.columns])
            
    # Replace NANs and NODATA with nans
    df = df.replace({" NAN": np.nan, " NODATA": np.nan})
    
    return df


def parse_par_header(file: str, attrs: Dict[str, dict]) -> Dict[str, dict]:
    """
    Parse the PAR (Photosynthetically Active Radiation) sensor header file.

    Parameters
    ----------
    file : str
        Path to PAR header file.
    attrs : dict
        Attribute metadata dictionary to update.

    Returns
    -------
    dict
        Updated attributes dictionary with PAR metadata.

    Notes
    -----
    Extracts instrument make, model, calibration date, serial number, and
    installation details from the PAR sensor header file.
    """
    with open(file) as f:
        for line in f:
            parsed = _parse_key_value(line)
            if parsed:
                key, value = parsed
                attrs["par"].update({key: value})
    return attrs


def parse_rad_header(file: str, attrs: Dict[str, dict]) -> Dict[str, dict]:
    """
    Parse the RAD (shortwave and longwave radiation) sensor header file.

    Parameters
    ----------
    file : str
        Path to RAD header file.
    attrs : dict
        Attribute metadata dictionary to update.

    Returns
    -------
    dict
        Updated attributes dictionary with RAD metadata.

    Notes
    -----
    Updates metadata for both shortwave and longwave sensors, including:
    - Instrument make and model
    - Calibration date
    - Serial numbers (SW and LW separately)
    - Installation details
    """
    with open(file) as f:
        header = f.readlines()

    for n, line in enumerate(header):
        if "make" in line.lower():
            key, value = _parse_key_value(line)
            _update_attrs(attrs, ["shortwave_radiation", "longwave_radiation"], key, value)
        elif "model" in line.lower():
            line = line.strip() + " " + header[n + 1].strip()
            key, value = _parse_key_value(line)
            _update_attrs(attrs, ["shortwave_radiation", "longwave_radiation"], key, value)
        elif "calibration date" in line.lower():
            key, value = _parse_key_value(line)
            _update_attrs(attrs, ["shortwave_radiation", "longwave_radiation"], key, value)
        elif "s/n" in line.lower():
            try:
                _, _, sn = line.split()
                if "swr" in line.lower():
                    attrs["shortwave_radiation"].update({"serial_number": sn})
                elif "lwr" in line.lower():
                    attrs["longwave_radiation"].update({"serial_number": sn})
            except ValueError:
                pass
        elif "installation" in line.lower():
            key, value = _parse_key_value(line)
            _update_attrs(attrs, ["shortwave_radiation", "longwave_radiation"], key, value)
    return attrs


def parse_met_header(file: str, attrs: Dict[str, dict]) -> Dict[str, dict]:
    """
    Parse the MET (meteorological) sensor header file.

    Parameters
    ----------
    file : str
        Path to MET header file (starboard = 'XTS', port = 'XTP').
    attrs : dict
        Attribute metadata dictionary to update.

    Returns
    -------
    dict
        Updated attributes dictionary with MET sensor metadata.

    Raises
    ------
    ValueError
        If the file is not recognized as port ('XTP') or starboard ('XTS').

    Notes
    -----
    Updates attributes such as:
    - Instrument make and model
    - Serial numbers (including PTU serials)
    - Calibration date
    - Installation date, location, and sensor height
    """
    if "XTS" in os.path.basename(file):
        keys = [x for x in attrs if "starboard" in x]
    elif "XTP" in os.path.basename(file):
        keys = [x for x in attrs if "port" in x]
    else:
        raise ValueError(f"Unrecognized MET file: {file}")

    with open(file) as f:
        for line in f:
            if "make" in line.lower() or "model" in line.lower():
                parsed = _parse_key_value(line)
                if parsed:
                    key, value = parsed
                    _update_attrs(attrs, keys, key, value)
            elif "s/n" in line.lower():
                _, sn, _, _, ptu = line.split()
                for k in keys:
                    attrs[k].update({"serial_number": sn.strip(",")})
                    attrs[k].update({"ptu_serial_number": ptu.strip(",")})
            elif any(word in line.lower() for word in ["calibration date", "installation date", "installation location", "height"]):
                parsed = _parse_key_value(line)
                if parsed:
                    key, value = parsed
                    _update_attrs(attrs, keys, key, value)
    return attrs


def parse_ssv_header(file: str, attrs: Dict[str, dict]) -> Dict[str, dict]:
    """
    Parse the SSV (Speed of Sound in Water) sensor header file.

    Parameters
    ----------
    file : str
        Path to SSV header file.
    attrs : dict
        Attribute metadata dictionary to update.

    Returns
    -------
    dict
        Updated attributes dictionary with SSV metadata.

    Notes
    -----
    Extracts:
    - Instrument type and model
    - Calibration date
    - Serial number
    - Installation date, location, and transducer distance
    """
    with open(file) as f:
        for line in f:
            parsed = _parse_key_value(line)
            if not parsed:
                continue
            key, value = parsed

            if "type" in line.lower():
                attrs["speed_of_sound"].update({key: value})
            elif "model" in line.lower():
                attrs["speed_of_sound"].update({key: value})
            elif "calibration date" in line.lower():
                attrs["speed_of_sound"].update({key: value})
            elif "s/n" in line.lower():
                attrs["speed_of_sound"].update({"serial_number": value})
            elif any(word in line.lower() for word in ["installation date", "installation location", "distance"]):
                attrs["speed_of_sound"].update({key: value})

    return attrs


def parse_sbe45_header(file: str, attrs: Dict[str, dict]) -> Dict[str, dict]:
    """
    Parse the SBE45 Thermosalinograph (TSG) header file.

    Parameters
    ----------
    file : str
        Path to SBE45 header file.
    attrs : dict
        Attribute metadata dictionary to update.

    Returns
    -------
    dict
        Updated attributes dictionary with SBE45 metadata.

    Notes
    -----
    Updates attributes for both:
    - Sea surface salinity
    - Sea surface temperature

    Metadata extracted includes:
    - Instrument name and model
    - Calibration date
    - Serial number
    - Installation date, location, and plumbing distance
    """
    with open(file) as f:
        for line in f:
            parsed = _parse_key_value(line)
            if not parsed:
                continue
            key, value = parsed

            if "instrument" in line.lower() or "model" in line.lower():
                attrs["sea_surface_salinity"].update({key: value})
                attrs["sea_surface_temperature"].update({key: value})
            elif "calibration date" in line.lower():
                attrs["sea_surface_salinity"].update({key: value})
                attrs["sea_surface_temperature"].update({key: value})
            elif "s/n" in line.lower():
                attrs["sea_surface_salinity"].update({"serial_number": value})
                attrs["sea_surface_temperature"].update({"serial_number": value})
            elif any(word in line.lower() for word in ["installation date", "installation location", "distance"]):
                attrs["sea_surface_salinity"].update({key: value})
                attrs["sea_surface_temperature"].update({key: value})

    return attrs


def parse_cruise_metadata(file: str) -> Dict[str, str]:
    """
    Parse the cruise metadata file.

    Parameters
    ----------
    file : str
        Path to cruise metadata file.

    Returns
    -------
    dict
        Metadata dictionary with key-value pairs extracted from the file.

    Notes
    -----
    Key names are normalized (lowercased, spaces replaced with underscores).
    The 'notes' section is parsed separately into a single consolidated string.
    """
    metadata: Dict[str, str] = {}
    with open(file) as f:
        lines = f.readlines()

    for n, line in enumerate(lines):
        if not line.strip():
            continue
        key, value = line.split(":", 1)
        key = "_".join(key.strip().lower().split())
        if "notes" in key.lower():
            metadata[key] = parse_notes(lines[n + 1 :])
            break
        metadata[key] = value.strip()
        
    return metadata


def parse_notes(lines: List[str]) -> str:
    """
    Parse multi-line notes from cruise metadata into a single string.

    Parameters
    ----------
    lines : list of str
        Lines from the cruise metadata file following the 'notes' section.

    Returns
    -------
    str
        Consolidated single-line notes string.
    """
    return " ".join(" ".join(l.split()).strip() for l in lines if l.strip())


def parse_ship_met_data(
    met_files: List[str],
    attrs: Dict[str, dict],
    met_headers: Optional[List[str]] = None,
    rad_header: Optional[str] = None,
    par_header: Optional[str] = None,
    sbe45_header: Optional[str] = None,
    ssv_header: Optional[str] = None,
    cruise_metadata: Optional[str] = None,
) -> xr.Dataset:
    """
    Parse ship meteorological data and associated header files into
    an xarray Dataset with standardized variable names and metadata.

    Parameters
    ----------
    met_files : list of str
        Paths to underway CSV data files.
    attrs : dict
        Attribute metadata dictionary (ATTRS) to update with header information.
    met_headers : list of str, optional
        Paths to MET sensor header files (port/starboard).
    rad_header : str, optional
        Path to radiation sensor header file.
    par_header : str, optional
        Path to PAR sensor header file.
    sbe45_header : str, optional
        Path to SBE45 thermosalinograph header file.
    ssv_header : str, optional
        Path to SSV (Speed of Sound) header file.
    cruise_metadata : str, optional
        Path to cruise metadata file.

    Returns
    -------
    xarray.Dataset
        Dataset containing parsed underway data with:
        - Variables renamed according to `name_map`
        - Attributes populated from instrument headers
        - Cruise-level metadata included in `ds.attrs` (if provided)

    Notes
    -----
    - All variables are cast to float.
    - Variable attributes (`.attrs`) are updated using the `ATTRS` dictionary.
    - Supports optional parsing of multiple instrument headers.
    - Intended for R/V Armstrong underway datasets but adaptable to similar ships.

    Examples
    --------
    >>> ds = parse_ship_met_data(
    ...     met_files=["met_data.csv"],
    ...     attrs=ATTRS,
    ...     met_headers=["WXTP_header.txt", "WXTS_header.txt"],
    ...     rad_header="RAD_header.txt",
    ...     par_header="PAR_header.txt",
    ...     cruise_metadata="cruise.metadata"
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions: ...
    Data variables:
        latitude   (time) float64 ...
        longitude  (time) float64 ...
        ...
    """
    # Parse met data
    data = parse_data_files(met_files)

    # Update attrs from headers
    if met_headers:
        for hdr in met_headers:
            attrs = parse_met_header(hdr, attrs)
    if rad_header:
        attrs = parse_rad_header(rad_header, attrs)
    if par_header:
        attrs = parse_par_header(par_header, attrs)
    if sbe45_header:
        attrs = parse_sbe45_header(sbe45_header, attrs)
    if ssv_header:
        attrs = parse_ssv_header(ssv_header, attrs)

    # Build Dataset
    ds = xr.Dataset(data)

    # Rename variables
    ds = ds.rename_vars({k: v for k, v in name_map.items() if k in ds})

    # Ensure float type
    for var in ds:
        ds[var] = ds[var].astype(float)

    # Add variable attributes
    for var in ds:
        if var in attrs:
            ds[var].attrs = attrs[var]

    # Add cruise metadata
    if cruise_metadata:
        ds.attrs = parse_cruise_metadata(cruise_metadata)

    return ds

