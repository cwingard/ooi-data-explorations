#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path

from setuptools import find_packages, setup

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# load version from `version.py` without importing the package
_version_ns = {}
with open(path.join(this_directory, 'version.py'), encoding='utf-8') as f:
    exec(f.read(), _version_ns)
version = _version_ns.get('__version__', _version_ns.get('version', '0.0.0'))

setup(
    name = 'ooi_data_explorations',
    version = version,
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
        'xarray[complete]',
        'numpy>=1.26',
        'packaging>24.1',
        'pandas>=2.2',
        'argcomplete',
        'beautifulsoup4',
        'canyonbpy',
        'cgsn_parsers @ https://bitbucket.org/ooicgsn/cgsn-parsers/get/master.zip',
        'cgsn_processing @ https://bitbucket.org/ooicgsn/cgsn-processing/get/master.zip',
        'distributed',
        'ephem',
        'erddapy',
        'gsw',
        'h5py',
        'ioos-qc',
        'ipykernel',
        'ipympl',
        'iris',
        'loguru',
        'munch',
        'nodejs',
        'nose',
        'numexpr',
        'pint',
        'ppigrf',
        'pyco2sys',
        'pyncml',
        'pysolar',
        'pyseas @ https://bitbucket.org/ooicgsn/pyseas/get/develop.zip',
        'PyYAML',
        's3fs',
        'scikit-learn',
        'sphinx',
        'tqdm',
        'xlrd'
    ],
    include_package_data=True,
    zip_safe=False
)
