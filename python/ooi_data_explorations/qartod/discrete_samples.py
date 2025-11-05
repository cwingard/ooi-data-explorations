#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Used to download discrete sample data from the OOI Alfresco WebDAV server
"""
import io
import numpy as np
import pandas as pd
import re
import requests

from bs4 import BeautifulSoup


def get_sites(url):
    """
    Get the list of arrays from the OOI Alfresco WebDAV server. The list is
    returned as a list of BeautifulSoup tags that can be further parsed to
    construct URLs to search for the different cruises per array that will
    have discrete sample data.

    :param url: OOI Alfresco WebDAV server URL
    :return sites: list of BeautifulSoup tags containing the array names
    """
    r = requests.get(url)
    if r.status_code == requests.codes.ok:
        soup = BeautifulSoup(r.text, 'html.parser')
        sites = soup.find_all('a', {'href': re.compile(
            r'^Argentine|Cabled|Endurance|Irminger|Pioneer|Southern|Station')})
        return sites
    else:
        return None


def get_cruises(url, sites, site_name, cruise=None):
    """
    Get the list of cruises for a specific array from the OOI Alfresco WebDAV
    server. The list is returned as a list of BeautifulSoup tags that can be
    further parsed to construct URLs to search for the different discrete
    sample data files per cruise.

    :param url: OOI Alfresco WebDAV server URL
    :param sites: list of BeautifulSoup tags containing the array names
    :param site_name: name of the array to search for
    :param cruise: optional, name of the cruise(s) to search for (can be
        either a single name or a list of names)
    :return cruises: list of BeautifulSoup tags containing the cruise name(s)
    """
    # if site_name is an empty string, return None (no site to search for)
    if not site_name:
        return None

    # find the site name in the list of sites
    site = [site for site in sites if site_name in site.text]
    if site is None:
        return None

    # create the URL for the site and search for the list of cruises
    a = re.findall(r'<a href=\"(.+?)\">', str(site))[0]
    site_url = url + a
    r = requests.get(site_url)
    if r.status_code == requests.codes.ok:
        soup = BeautifulSoup(r.text, 'html.parser')
        if cruise:
            if isinstance(cruise, list):
                cruises = [soup.find_all('a', {'href': re.compile('.*' + c + '.+?')})[0]
                           for c in cruise]
            else:
                cruises = soup.find_all('a', {'href': re.compile('.*' + cruise + '.+?')})
        else:
            if 'Pioneer' in site_name:
                # the Pioneer array is split into two sites, but the cruise names are not
                cruises = soup.find_all('a', {'href': re.compile('Pioneer')})
            else:
                cruises = soup.find_all('a', {'href': re.compile(site_name)})

        return cruises
    else:
        return None


def get_discrete_samples(site_name, cruise=None):
    """
    Get the discrete sample data for a specific array and cruise from the OOI
    Raw Data server. The data is returned as a pandas DataFrame.

    :param site_name: name of the array to search for
    :param cruise: optional, name of the cruise(s) to search for (can be
        either a single name or a list of names)
    :return samples: pandas DataFrame containing the discrete sample data
    """
    # check that the site name is a valid string
    if not isinstance(site_name, str) or not site_name:
        raise ValueError("site_name must be a non-empty string")

    # check that the site name is a valid OOI array name
    valid_sites = ['Argentine_Basin', 'Cabled', 'Endurance', 'Irminger_Sea', 'Pioneer-MAB',
                   'Pioneer-NES', 'Southern_Ocean', 'Station_Papa']
    if site_name not in valid_sites:
        raise ValueError(f"site_name must be one of {valid_sites}")

    # check that the cruise is a valid string or list of strings
    if cruise is not None:
        if isinstance(cruise, str):
            cruise = [cruise]
        elif not isinstance(cruise, list) or not all(isinstance(c, str) for c in cruise):
            raise ValueError("cruise must be a string or a list of strings")

    # base URL for the OOI Raw Data server where the discrete sample data is stored
    base_url = 'https://rawdata.oceanobservatories.org/files/cruise_data/'

    # create a listing of the array names in the WebDAV directory
    sites = get_sites(base_url)

    # use the site name to find the URLs for the different cruises at that site
    cruises = get_cruises(base_url, sites, site_name, cruise)
    if cruises is None:
        # could not find the specified site or cruise(s)
        return None

    # use the list of cruises to find the URLs for the discrete sample CSVs for each cruise and
    # download the data into a pandas DataFrame
    samples = []
    for sample in cruises:
        # create the URL for the cruise discrete sample data directory
        sample_url = base_url + site_name + '/' +re.findall(r'<a href=\"(.+?)\">', str(sample))[0] + 'Water_Sampling/'

        # search for the discrete sample data files (note, not all cruises have discrete sample data files)
        r = requests.get(sample_url)
        if r.status_code != requests.codes.ok:
            continue

        # find the CSV file for the discrete sample data and create the URL to download the file
        soup = BeautifulSoup(r.text, 'html.parser')
        csv = soup.find_all('a', {'href': re.compile(r'.+?Discrete_Summary\.csv')})
        if not csv:
            continue
        else:
            csv = csv[0]
        csv_url = sample_url + re.findall(r'<a href=\"(.+?)\">', str(csv))[0]

        # download the CSV file and read it into a pandas DataFrame appending it to the list of discrete
        # sample data files from the different cruises, if any
        r = requests.get(csv_url)
        if r.status_code == requests.codes.ok:
            times = ['Start Time [UTC]', 'CTD Bottle Closure Time [UTC]']
            samples.append(pd.read_csv(io.StringIO(r.content.decode('utf-8')), parse_dates=times, date_format='ISO8601'))

    if not samples:
        # could not find any discrete sample data files for the specified site and cruise(s)
        return None

    # concatenate the list of DataFrames into a single DataFrame
    samples = pd.concat(samples, ignore_index=True)

    # replace the fill values with NaNs for all sample columns (ignoring the flag columns
    # and a few other select columns)
    for column in samples.columns:
        if 'Flag' in column:
            # convert the flag columns to a string type
            samples[column] = samples[column].astype(str)
        else:
            if column not in ['Cruise', 'Station', 'Target Asset', 'Cast', 'CTD File',
                              'CTD Bottle Closure Time [UTC]']:
                samples[column] = samples[column].replace(-9999999., value=np.nan)

    return samples


def distance_to_cast(samples, lat, lon):
    """
    Calculate the distance to the CTD cast (or other sampling method) location
    from the specified lat/lon coordinates (usually the mooring) using the
    Haversine formula.

    :param samples: pandas DataFrame containing the discrete sample data
    :param lat: latitude of the location to calculate the distance from
    :param lon: longitude of the location to calculate the distance from
    :return distance: pandas Series containing the distance to the cast location
    """
    r = 6371.0  # radius of the Earth in km (assumes a spherical Earth)
    mlat = np.radians(lat)                                  # mooring latitude
    slat = np.radians(samples['Start Latitude [degrees]'])  # sample latitudes

    # calculate the differences in latitude and longitude
    dlat = np.radians(lat - samples['Start Latitude [degrees]'])
    dlon = np.radians(lon - samples['Start Longitude [degrees]'])

    # calculate the distance using the Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(mlat) * np.cos(slat) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # calculate the distance in km
    distance = r * c
    return distance
