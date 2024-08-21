#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from ooi_data_explorations.common import m2m_request, m2m_collect, load_gc_thredds, get_deployment_dates, \
    get_vocabulary, update_dataset, CONFIG, ENCODINGS
from ooi_data_explorations.uncabled.process_adcp import adcp_instrument

# Setup needed parameters for the request, the user would need to vary
# these to suit their own needs and sites/instruments of interest. Site,
# node, sensor, stream and delivery method names can be obtained from the
# Ocean Observatories Initiative website. The last two parameters (level
# and instrmt) will set path and naming conventions to save the data to the
# local disk.
site = 'CE02SHSM'           # OOI Net site designator
node = 'RID26'              # OOI Net node designator
sensor = '01-ADCPTA000'     # OOI Net sensor designator
method = 'telemetered'      # OOI Net data delivery method
level = 'midwater'          # local directory name, level below site
instrmt = 'adcp'            # local directory name, instrument below level

# download some of the data for deployment 17 from the Gold Copy THREDDS catalog, the rest from the M2M system
tag = 'deployment0017.*ADCP.*\\.nc$'
earth = load_gc_thredds(site, node, sensor, method, 'adcp_velocity_earth', tag)

# the so-called engineering data are only available from the M2M system because they are not considered
# "science" data. The false distinction between science and engineering data is a legacy of earlier OOI data
# management practices that mistakenly assigned meaning and value to the data prior to its use.
start, stop = get_deployment_dates(site, node, sensor, 17)
m = m2m_request(site, node, sensor, method, 'adcp_engineering', start, stop)
engineering = m2m_collect(m, tag)

# clean-up and reorganize the data
adcp = adcp_instrument(earth, engineering)
vocab = get_vocabulary(site, node, sensor)[0]
adcp = update_dataset(adcp, vocab['maxdepth'])

# save the data
out_path = os.path.join(CONFIG['base_dir']['m2m_base'], site.lower(), level, instrmt)
out_path = os.path.abspath(out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

out_file = ('%s.%s.%s.deploy%02d.%s.%s.nc' % (site.lower(), level, instrmt, 17, method, 'adcp_velocity_earth'))
nc_out = os.path.join(out_path, out_file)

adcp.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)

