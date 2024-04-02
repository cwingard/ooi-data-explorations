#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from ooi_data_explorations.common import get_vocabulary, load_gc_thredds, update_dataset, CONFIG, ENCODINGS
from ooi_data_explorations.uncabled.process_vel3d import mmp_aquadopp

# Setup needed parameters for the request, the user would need to vary
# these to suit their own needs and sites/instruments of interest. Site,
# node, sensor, stream and delivery method names can be obtained from the
# Ocean Observatories Initiative website. The last two parameters (level
# and instrmt) will set path and naming conventions to save the data to the
# local disk.
site = 'CE09OSPM'           # OOI Net site designator
node = 'WFP01'              # OOI Net node designator
sensor = '01-VEL3DK000'     # OOI Net sensor designator
method = 'telemetered'    # OOI Net data delivery method
stream = 'vel3d_k_wfp_stc_instrument'  # OOI Net stream name
level = 'mmp'               # local directory name, level below site
instrmt = 'aquadopp'        # local directory name, instrument below level

# get the vocabulary for the site and the site depth
vocab = get_vocabulary(site, node, sensor)[0]
site_depth = vocab['maxdepth']

# download data for deployment 17 from the Gold Copy THREDDS catalog
tag = 'deployment0017.*VEL3D.*\\.nc$'
ds = load_gc_thredds(site, node, sensor, method, stream, tag)

# clean-up and reorganize the data, binning the data to 2 m depth intervals
aquadopp = mmp_aquadopp(ds, binning=True, bin_size=2.0)
aquadopp = update_dataset(aquadopp, site_depth)

# save the data
out_path = os.path.join(CONFIG['base_dir']['m2m_base'], site.lower(), level, instrmt)
out_path = os.path.abspath(out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

out_file = ('%s.%s.%s.deploy17.%s.%s.nc' % (site.lower(), level, instrmt, method, 'mmp_aquadopp_data_telemetered'))
nc_out = os.path.join(out_path, out_file)
aquadopp.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)
