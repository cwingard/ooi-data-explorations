#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from ooi_data_explorations.common import list_deployments, get_deployment_dates, get_vocabulary, \
    load_gc_thredds, update_dataset, CONFIG, ENCODINGS
from ooi_data_explorations.uncabled.process_metbk import metbk_datalogger

# Setup needed parameters for the request, the user would need to vary these to suit their own needs and
# sites/instruments of interest. Site, node, sensor, stream and delivery method names can be obtained from the
# Ocean Observatories Initiative web site. The last two parameters (level and instrmt) will set path and naming
# conventions to save the data to the local disk.
site = 'CE02SHSM'           # OOI Net site designator
node = 'SBD11'              # OOI Net node designator
sensor = '06-METBKA000'     # OOI Net sensor designator
method = 'telemetered'      # OOI Net data delivery method
stream = 'metbk_a_dcl_instrument'  # OOI Net stream name
level = 'buoy'              # local directory name, level below site
instrmt = 'metbk'           # local directory name, instrument below level

# Load all telemetered data collected to date for the specified METBK instrument
metbk = load_gc_thredds(site, node, sensor, method, stream, r'.*METBK.*\.nc$')

# clean-up and reorganize the METBK data set
metbk = metbk_datalogger(metbk)
metbk = update_dataset(metbk, 0.0)

# save the data -- utilize groups for the metbk and water datasets
out_path = os.path.join(CONFIG['base_dir']['m2m_base'], site.lower(), level, instrmt)
out_path = os.path.abspath(out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

out_file = ('%s.%s.%s..%s.%s.combined.nc' % (site.lower(), level, instrmt, method, stream))
nc_out = os.path.join(out_path, out_file)
metbk.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)
