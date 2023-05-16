#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from ooi_data_explorations.common import list_deployments, get_vocabulary, load_gc_thredds, \
    update_dataset, CONFIG, ENCODINGS
from ooi_data_explorations.cabled.process_optaa import optaa_profiler


def main():
    # Setup needed parameters for the request, the user would need to vary
    # these to suit their own needs and sites/instruments of interest. Site,
    # node, sensor, stream and delivery method names can be obtained from the
    # Ocean Observatories Initiative website. The last two parameters (level
    # and instrmt) will set path and naming conventions to save the data to the
    # local disk.
    site = 'CE04OSPS'                             # OOI Net site designator
    node = 'SF01B'                                # OOI Net node designator
    sensor = '3B-OPTAAD105'                       # OOI Net sensor designator
    stream = 'optaa_sample'                       # OOI Net stream name
    method = 'streamed'                           # OOI Net data delivery method
    level = 'profiler'                            # local directory name, level below site
    instrmt = 'optaa'                             # local directory name, instrument below level

    # We are after streamed data from deployment 6
    vocab = get_vocabulary(site, node, sensor)[0]
    deploy = 6
    tag = '.*deployment{:04d}.*OPTAA.*\\.nc$'.format(deploy)  # download OPTAA files from the current deployment

    # OPTAA data is best downloaded from the Gold Copy THREDDS catalog (much faster than an M2M request)
    optaa = load_gc_thredds(site, node, sensor, method, stream, tag)

    # set up the calibration file path and name
    cal_path = os.path.join(os.path.expanduser('~'), 'ooidata/m2m', site.lower(), level, instrmt)
    cal_path = os.path.abspath(cal_path)
    if not os.path.exists(cal_path):
        os.makedirs(cal_path)

    cal_file = ('{}.{}.{}.deploy{:02d}.cal_coeffs.json'.format(site.lower(), level, instrmt, deploy))
    cal_file = os.path.join(cal_path, cal_file)

    # clean-up and reorganize
    optaa = optaa_profiler(optaa, cal_file)
    optaa = update_dataset(optaa, vocab['maxdepth'])

    # save the data
    out_path = os.path.join(CONFIG['base_dir']['m2m_base'], site.lower(), level, instrmt)
    out_path = os.path.abspath(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_file = ('%s.%s.%s.deploy%02d.%s.%s.nc' % (site.lower(), level, instrmt, deploy, method, stream))
    nc_out = os.path.join(out_path, out_file)

    optaa.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)


if __name__ == '__main__':
    main()
