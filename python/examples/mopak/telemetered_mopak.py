#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from ooi_data_explorations.common import ENCODINGS, list_deployments, get_deployment_dates, get_vocabulary, \
    m2m_request, m2m_collect, update_dataset, dict_update
from ooi_data_explorations.uncabled.process_mopak import mopak_datalogger

# Setup needed parameters for the request, the user would need to vary these to
# suit their own needs and sites/instruments of interest. Site, node, sensor,
# stream and delivery method names can be obtained from the Ocean Observatories
# Initiative website. The last two will set path and naming conventions to save
# the data to the local disk
site = 'CE02SHSM'                               # OOI Net site designator
node = 'SBD11'                                  # OOI Net node designator
sensor = '01-MOPAK0000'                         # OOI Net sensor designator
stream = 'mopak_o_dcl_accel'                    # OOI Net stream name
method = 'telemetered'                          # OOI Net data delivery method
level = 'buoy'                                  # local directory name, level below site
instrmt = 'mopak'                               # local directory name, instrument below level

# We are after telemetered data. Determine list of deployments and use the last, presumably currently active,
# deployment to determine the start and end dates for our request.
deployments = list_deployments(site, node, sensor)
deploy = deployments[-1]
start, end = get_deployment_dates(site, node, sensor, deploy)

# request the data from the OOI system
tag = '.*deployment{:04d}.*MOPAK.*\\.nc$'.format(deploy)  # download MOPAK files from the current deployment
r = m2m_request(site, node, sensor, method, stream, start, end)
mopak = m2m_collect(r, tag)

# get the vocabulary information for the site, node, sensor
vocab = get_vocabulary(site, node, sensor)[0]

# clean-up and reorganize
mopak, waves = mopak_datalogger(mopak)

# set up compression of the MOPAK data variables to save on file size (note, these can get rather large, consider
# splitting the data into smaller files if needed to avoid file size limitations)
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in mopak.data_vars}

# update the dataset to better match CF conventions
mopak = update_dataset(mopak, vocab['maxdepth'])
waves = update_dataset(waves, vocab['maxdepth'])

# save the data to local disk
out_path = os.path.join(os.path.expanduser('~'), 'ooidata/m2m', site.lower(), level, instrmt)
out_path = os.path.abspath(out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

out_file = ('{}.{}.{}.deploy{:02d}.{}.{}.nc'.format(site.lower(), level, instrmt, deploy, method, stream))
nc_out = os.path.join(out_path, out_file)
mopak.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf', encoding=dict_update(ENCODINGS, encoding))

out_file = ('{}.{}.{}.deploy{:02d}.{}.{}_waves.nc'.format(site.lower(), level, instrmt, deploy, method, stream))
nc_out = os.path.join(out_path, out_file)
waves.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)
