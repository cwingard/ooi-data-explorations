#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path

from setuptools import find_packages, setup

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name = 'ooi_data_explorations',
    version = '0.2.3',
    description = (
        'Collection of python processing modules for requesting data '
        'from the OOI M2M system'
    ),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    classifiers = [
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: ONLY',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    keywords = [
        'Ocean Observatories Initiative', 'Regional Cabled Array',
        'Coastal Endurance Array', 'Coastal Pioneer Array',
        'Global Papa Array', 'Global Irminger Array'
    ],
    url = 'https://github.com/oceanobservatories/ooi-data-explorations/python/',
    author = 'Christopher Wingard',
    author_email = 'chris.wingard@oregonstate.edu',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'xarray[accel]',
        'dask',
        'munch',
        'tqdm',
        'urllib3',
        'numpy',
        'pandas',
        'gsw',
        'requests',
        'beautifulsoup4',
        'PyYAML',
        'matplotlib',
        'ntplib',
        'netcdf4',
        'h5netcdf',
    ],
    include_package_data=True,
    zip_safe=False
)
