#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from ooi_data_explorations.common import load_gc_thredds, update_dataset, CONFIG, ENCODINGS
from ooi_data_explorations.uncabled.process_velpt import velpt_datalogger


def main():
    # Setup needed parameters for the request, the user would need to vary
    # these to suit their own needs and sites/instruments of interest. Site,
    # node, sensor, stream and delivery method names can be obtained from the
    # Ocean Observatories Initiative web site. The last two parameters (level
    # and instrmt) will set path and naming conventions to save the data to the
    # local disk.
    site = 'CE02SHSM'           # OOI Net site designator
    node = 'SBD11'              # OOI Net node designator
    sensor = '04-VELPTA000'     # OOI Net sensor designator
    stream = 'velpt_ab_dcl_instrument_recovered'  # OOI Net stream name
    method = 'recovered_host'   # OOI Net data delivery method
    level = 'buoy'              # local directory name, level below site
    instrmt = 'velpt'           # local directory name, instrument below level

    # download the data for deployment 15 from the Gold Copy THREDDS catalog
    velpt = load_gc_thredds(site, node, sensor, method, stream, 'deployment0015.*VELPT.*\\.nc$')

    # clean-up and reorganize
    velpt = velpt_datalogger(velpt)
    depth = velpt['depth'].mean().values
    velpt = update_dataset(velpt, depth)

    # save the data
    out_path = os.path.join(CONFIG['base_dir']['m2m_base'], site.lower(), level, instrmt)
    out_path = os.path.abspath(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_file = ('%s.%s.%s.deploy17.%s.%s.nc' % (site.lower(), level, instrmt, method, stream))
    nc_out = os.path.join(out_path, out_file)

    velpt.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)


if __name__ == '__main__':
    main()
