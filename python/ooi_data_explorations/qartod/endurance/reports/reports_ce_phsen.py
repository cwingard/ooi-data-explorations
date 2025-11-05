#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the PHSEN data from the uncabled, Coastal Endurance Surface
    Moorings and processes the data to generate report plots and tables for
    HITL QC review.
"""
import dateutil.parser as parser
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

from matplotlib.dates import DateFormatter

from ooi_data_explorations.common import get_annotations, get_vocabulary, get_deployment_dates, load_gc_thredds
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.uncabled.process_phsen import phsen_datalogger, phsen_instrument
from ooi_data_explorations.qartod.discrete_samples import get_discrete_samples, distance_to_cast
from ooi_data_explorations.qartod.qc_processing import ANNO_HEADER, inputs
from ooi_data_explorations.qartod.reporting import apply_qc_results


def combine_delivery_methods(site, node, sensor, deployment):
    """
    Downloads the data from each of the three data delivery methods for
    the uncabled pH (PHSEN) instrument for a specific deployment and
    returns a list object containing the data from each delivery method.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :param deployment: deployment number to download the data for
    :return merged: the three PHSEN data streams as a list object
    """
    # set the regex tag for the data files to download
    tag = 'deployment{:04d}.*PHSEN.*\\.nc$'.format(deployment)
    print('### Downloading data for {}-{}-{}, deployment {:02d} ###'.format(site, node, sensor, deployment))
    warnings.filterwarnings("error")  # set warnings to raise exceptions, so we can catch them
    # download the data from the telemetered data stream
    try:
        telem = load_gc_thredds(site, node, sensor, 'telemetered', 'phsen_abcdef_dcl_instrument', tag)
    except UserWarning:
        telem = None
    # download the data from the recovered_host data stream
    try:
        rhost = load_gc_thredds(site, node, sensor, 'recovered_host', 'phsen_abcdef_dcl_instrument_recovered', tag)
    except UserWarning:
        rhost = None
    # download the data from the recovered_inst data stream
    try:
        rinst = load_gc_thredds(site, node, sensor, 'recovered_inst', 'phsen_abcdef_instrument', tag)
    except UserWarning:
        rinst = None
    # reset the warnings to default behavior
    warnings.resetwarnings()

    # combine the data from the three delivery methods
    return [telem, rhost, rinst]


def plotting(site, node, sensor, deployment, dates, availability, merged, qartod, pre_merged, post_merged, samples):
    """
    Generate plots of the PHSEN data for the telemetered, recovered host and
    recovered instrument data streams, as well as the pre- and post-deployment
    data for the sensor.
    """
    params = {
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.titlelocation': 'right',
        'figure.figsize': [11, 17],
        'figure.dpi': 100,
        'xtick.major.size': 4,
        'xtick.major.pad': 4,
        'xtick.major.width': 1.0,
        'ytick.major.size': 4,
        'ytick.major.pad': 4,
        'ytick.major.width': 1.0
    }
    plt.rcParams.update(params)

    # initialize the report figure and set the title
    fig = plt.figure(layout='tight')
    gs = fig.add_gridspec(9, 2)
    fig.subplots_adjust(top=0.90)
    fig.suptitle('{}-{}-{}, Deployment {:02d}'.format(site.upper(), node.upper(), sensor.upper(), deployment), y=0.99)

    # get the maximum depth for the sensor
    vocab = get_vocabulary(site, node, sensor)[0]
    depth = vocab['maxdepth']

    # create the plots for each parameter
    ax1 = fig.add_subplot(gs[0, :])
    labels = ['T', 'RH', 'RI']
    colors = ['RoyalBlue', 'DarkOrange', 'ForestGreen']
    yvalues = [1.3, 1, 0.7]
    for i in range(len(availability)):
        if availability[i] is not None:
            ax1.plot(availability[i], availability[i].astype(int) * 0 + yvalues[i], 'o', label=labels[i],
                     color=colors[i])
    ax1.set_xlim(dates)
    ax1.set_ylim(0.5, 1.50)
    ax1.set_yticks(yvalues)
    ax1.set_yticklabels(labels)

    ax2 = fig.add_subplot(gs[1:3, :])
    ax2.plot(merged.time, merged.seawater_ph, label='Active', color='RoyalBlue')
    ax2.set_xlim(dates)
    ax2.set_ylim(6.9, 9.0)
    ax2.set_yticks([7.0, 7.5, 8.0, 8.5, 9.0])
    ax2.set_ylabel('Sea Water pH')

    ax3 = fig.add_subplot(gs[3:5, 0])
    zoom = [dates[0] - pd.Timedelta('15D'), dates[0] + pd.Timedelta('15D')]
    ax3.plot(merged.time, merged.seawater_ph, label='Active', color='RoyalBlue')
    if pre_merged is not None:
        ax3.plot(pre_merged.time, pre_merged.seawater_ph, label='Pre', color='grey')
    ax3.scatter(samples['Start Time [UTC]'], samples['Calculated pH'], marker='*', label='Discrete',
                color='DarkOrange')
    ax3.scatter(samples['Start Time [UTC]'], samples['CTD pH'], marker='+', label='CTD pH',
                color='ForestGreen')
    ax3.set_xlim(zoom)
    x_fmt = DateFormatter('%b-%d')
    ax3.xaxis.set_major_formatter(x_fmt)
    ax3.set_xlabel('')
    ax3.set_ylim(6.9, 9.0)
    ax3.set_yticks([7.0, 7.5, 8.0, 8.5, 9.0])
    ax3.set_ylabel('Sea Water pH')
    ax3.legend(loc='lower left', fontsize='small')

    ax4 = fig.add_subplot(gs[3:5, 1])
    zoom = [dates[1] - pd.Timedelta('15D'), dates[1] + pd.Timedelta('15D')]
    ax4.plot(merged.time, merged.seawater_ph, label='Active', color='RoyalBlue')
    if post_merged is not None:
        ax4.plot(post_merged.time, post_merged.seawater_ph, label='Post', color='grey')
    ax4.scatter(samples['Start Time [UTC]'], samples['Calculated pH'], marker='*', label='Discrete',
                color='DarkOrange')
    ax4.scatter(samples['Start Time [UTC]'], samples['CTD pH'], marker='+', label='CTD pH',
                color='ForestGreen')
    ax4.set_xlim(zoom)
    ax4.xaxis.set_major_formatter(x_fmt)
    ax4.set_xlabel('')
    ax4.set_ylim(6.9, 9.0)
    ax4.set_yticks([7.0, 7.5, 8.0, 8.5, 9.0])
    ax4.legend(loc='lower left', fontsize='small')

    ax5 = fig.add_subplot(gs[5, :])
    ax5.plot(merged.time, merged.signal_434, label='434 nm', color='RoyalBlue')
    ax5.set_xlim(dates)
    ax5.set_ylim(0, 4100)
    ax5.set_ylabel('Signal')
    ax5.legend(['434 nm'], loc='upper left', fontsize='small')

    ax6 = fig.add_subplot(gs[6, :])
    ax6.plot(merged.time, merged.signal_578, label='578 nm', color='DarkOrange')
    ax6.set_xlim(dates)
    ax6.set_ylim(0, 4100)
    ax6.set_ylabel('Signal')
    ax6.legend(['578 nm'], loc='upper left', fontsize='small')

    ax7 = fig.add_subplot(gs[7, :])
    ax7.plot(merged.time, merged.reference_434, label='434 nm', color='RoyalBlue')
    ax7.plot(merged.time, merged.reference_578, label='578 nm', color='DarkOrange')
    ax7.set_xlim(dates)
    ax7.set_ylim(0, 4100)
    ax7.set_ylabel('Blanks')

    ax8 = fig.add_subplot(gs[8, :], sharex=ax1)
    auto = qartod[0]
    self = qartod[1]
    anno = qartod[2]
    ax8.plot([zoom[1], zoom[1], zoom[1], zoom[1], zoom[1], zoom[1]],
             ['0', '1', '2', '3', '4', '9'], '.', color='grey', label='_nolegend_')
    ax8.set_yticks(['0', '1', '2', '3', '4', '9'])  # need to create a dummy plot to set the y-ticks
    ax8.set_ylabel('Flags')
    ax8.plot(auto[0].time, auto[0].values.astype(str), '*', label='QARTOD', color='RoyalBlue')
    ax8.plot(self[0].time, self[0].values.astype(str), '+', label='Self', color='DarkOrange')
    if len(anno) > 0:
        ax8.plot(anno[0].time, anno[0].values.astype(str), '.', label='Anno', color='DarkGreen')
    ax8.legend(loc='best', fontsize='small')

    # return the figure object
    return fig


def generate_report(site, node, sensor, deployment):
    """
    Load the CTD data for a defined reference designator (using the site, node
    and sensor names to construct the reference designator) collected via the
    telemetered, recovered host and instrument methods and use the data to
    generate a report for HITL QC review.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :param deployment: deployment number to download the data for

    :return merged: the merged PHSEN data streams for the deployment
    :return fig: the report figure object for the deployment
    :return annotations: the annotations for the sensor for the deployment
    """
    # load the data from the three delivery methods
    data = combine_delivery_methods(site, node, sensor, deployment)

    # also grab the data from the prior and post deployments
    pre_data = combine_delivery_methods(site, node, sensor, deployment - 1)
    post_data = combine_delivery_methods(site, node, sensor, deployment + 1)

    # create an availability data array for each data stream
    availability = [None, None, None]
    for i in range(len(data)):
        if data[i] is not None:
            availability[i] = data[i].time

    # get the start and end dates for the deployment
    start, end = get_deployment_dates(site, node, sensor, deployment)
    if start is not None and end is not None:
        start = parser.parse(start, ignoretz=True)
        end = parser.parse(end, ignoretz=True)
        dates = [start, end]
    else:
        raise ValueError('Deployment dates for {}-{}-{} deployment {:02d} not found'.format(site, node, sensor, deployment))

    # get the current system annotations for the sensor
    annotations = get_annotations(site, node, sensor)
    annotations = pd.DataFrame(annotations)
    if not annotations.empty:
        annotations = annotations.drop(columns=['@class'])
        annotations['beginDate'] = pd.to_datetime(annotations.beginDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
        annotations['endDate'] = pd.to_datetime(annotations.endDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Convert the text based QC flags to numeric QARTOD-style flags
        codes = {
            None: 0,
            'pass': 1,
            'suspect': 3,
            'fail': 4,
            'not_operational': 9,
            'not_available': 9
        }
        annotations['qcFlag'] = annotations['qcFlag'].map(codes).astype('category')

        # limit the annotations to the deployment dates making sure to catch any that might span the deployment, be
        # entirely within the deployment, or have start and/or end dates that fall within the deployment dates
        annotations = annotations[((annotations.beginDate <= start.strftime('%Y-%m-%dT%H:%M:%S')) &
                                   (annotations.endDate >= end.strftime('%Y-%m-%dT%H:%M:%S'))) |
                                  ((annotations.beginDate >= start.strftime('%Y-%m-%dT%H:%M:%S')) &
                                   (annotations.endDate <= end.strftime('%Y-%m-%dT%H:%M:%S'))) |
                                  ((annotations.beginDate <= start.strftime('%Y-%m-%dT%H:%M:%S')) &
                                   (annotations.endDate >= start.strftime('%Y-%m-%dT%H:%M:%S')) &
                                   (annotations.endDate <= end.strftime('%Y-%m-%dT%H:%M:%S'))) |
                                  ((annotations.beginDate >= start.strftime('%Y-%m-%dT%H:%M:%S')) &
                                   (annotations.beginDate <= end.strftime('%Y-%m-%dT%H:%M:%S')) &
                                   (annotations.endDate >= end.strftime('%Y-%m-%dT%H:%M:%S')))]

        # sort the annotations by the beginDate
        annotations = annotations.sort_values(by='beginDate')

    # working through the 3 data streams, reformat the data, add the annotations and apply the QC tests
    for i in range(len(data)):
        if i < 2:
            if data[i]:
                data[i] = phsen_datalogger(data[i])
                data[i] = apply_qc_results(data[i], annotations)
            if pre_data[i]:
                pre_data[i] = phsen_datalogger(pre_data[i])
                pre_data[i] = apply_qc_results(pre_data[i], annotations)
            if post_data[i]:
                post_data[i] = phsen_datalogger(post_data[i])
                post_data[i] = apply_qc_results(post_data[i], annotations)
        else:
            if data[i]:
                data[i] = phsen_instrument(data[i])
                data[i] = apply_qc_results(data[i], annotations)
            if pre_data[i]:
                pre_data[i] = phsen_instrument(pre_data[i])
                pre_data[i] = apply_qc_results(pre_data[i], annotations)
            if post_data[i]:
                post_data[i] = phsen_instrument(post_data[i])
                post_data[i] = apply_qc_results(post_data[i], annotations)

    # combine the data from the three delivery methods into a single dataset resampled to 1 hour intervals
    merged = combine_datasets(data[0], data[1], data[2], 60)

    # re-work the QC variables to ensure they are in the correct format (integer) and fill in any missing values
    # with the default value of 9 (missing data)
    for var in merged.variables:
        if var == 'seawater_ph_quality_flag':
            merged['seawater_ph_quality_flag'] = merged['seawater_ph_quality_flag'].where(
                ~np.isnan(merged['seawater_ph_quality_flag']), 9).astype(int)

        if var == 'seawater_ph_qartod_results':
            merged['seawater_ph_qartod_results'] = merged['seawater_ph_qartod_results'].where(
                ~np.isnan(merged['seawater_ph_qartod_results']), 9).astype(int)

        if var == 'rollup_annotations_qc_results':
            merged['rollup_annotations_qc_results'] = merged['rollup_annotations_qc_results'].where(
                ~np.isnan(merged['rollup_annotations_qc_results']), 9).astype(int)

    # pull out the QC results for the 4 parameters of interest
    qartod = [
        [merged[x].sortby(merged[x]) for x in merged.variables if 'qartod_results' in x],
        [merged[x].sortby(merged[x]) for x in merged.variables if 'ph_quality_flag' in x],
        [merged[x].sortby(merged[x]) for x in merged.variables if 'annotations_qc_results' in x]
    ]

    # combine the pre- and post- data from the three delivery methods into a single dataset (pre- and post- data are
    # really used just for plotting purposes) trimmed to a +- 15-day window around the deployment dates
    pre_merged = combine_datasets(pre_data[0], pre_data[1], pre_data[2], None)
    if pre_merged is not None:
        pre_merged = pre_merged.sel(time=slice(start - pd.Timedelta('15D'), start + pd.Timedelta('15D')))
        if pre_merged.time.size > 0:
            pre_merged['time'] = pre_merged['time'] + pd.Timedelta('30Min')
            pre_merged = pre_merged.resample(time='1h', skipna=True).median(dim='time', keep_attrs=True)
            pre_merged = pre_merged.where(~np.isnan(pre_merged.deployment), drop=True)
        else:
            pre_merged = None

    post_merged = combine_datasets(post_data[0], post_data[1], post_data[2], None)
    if post_merged is not None:
        post_merged = post_merged.sel(time=slice(end - pd.Timedelta('15D'), end + pd.Timedelta('15D')))
        if post_merged.time.size > 0:
            post_merged['time'] = post_merged['time'] + pd.Timedelta('30Min')
            post_merged = post_merged.resample(time='1h', skipna=True).median(dim='time', keep_attrs=True)
            post_merged = post_merged.where(~np.isnan(post_merged.deployment), drop=True)
        else:
            post_merged = None

    # load the discrete sample data for the deployment cruise
    samples = get_discrete_samples('Endurance')

    # limit the discrete samples to those collected within 3 meters of the instrument depth
    vocab = get_vocabulary(site, node, sensor)[0]
    depth = vocab['maxdepth']
    samples = samples[(samples['CTD Depth [m]'] - depth).abs() <= 3.0]

    # determine the distance to the sampling location for each discrete sample relative to the mooring location
    distance = distance_to_cast(samples, merged.attrs['lat'], merged.attrs['lon'])

    # use the distance to the cast to determine the closest samples to the mooring location within a defined distance
    dmin = 1.0  # 1.0 km for the shallow, inshore sites
    if site in ['CE02SHSM', 'CE07SHSM']:
        dmin = 3.0  # 3.0 km for the shelf sites
    if site in ['CE04OSSM', 'CE09OSSM']:
        dmin = 5.0  # 5.0 km for the offshore sites

    samples = samples[distance <= dmin]

    # generate the report plot
    fig = plotting(site, node, sensor, deployment, dates, availability, merged, qartod,
                   pre_merged, post_merged, samples)

    # return the merged data, figure object and the annotations
    return merged, fig, annotations


def main(argv=None):
    """
    Download the PHSEN data from the Gold Copy THREDDS server and create a
    plot page and annotations table for subsequent HITL QC reviews.
    """
    # set up the input arguments
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    deployment = args.deployment

    # save the downloaded annotations and plots to a local directory
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/reports')
    platform = 'midwater'
    if node == 'MFD35':
        platform = 'seafloor'

    out_path = os.path.join(out_path, site, platform, 'phsen')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # generate the report data, plot and annotations
    data, fig, annotations = generate_report(site, node, sensor, deployment)

    # save the data to a netCDF file for further processing
    nc_file = '{}-{}-{}.Deploy{:02d}.merged_data.nc'.format(site, node, sensor, deployment)
    data.to_netcdf(os.path.join(out_path, nc_file), format='NETCDF4', engine='h5netcdf')

    # save the annotations to a csv file for further processing
    anno_csv = '{}-{}-{}.Deploy{:02d}.quality_annotations.csv'.format(site, node, sensor, deployment)
    annotations.to_csv(os.path.join(out_path, anno_csv), index=False, columns=ANNO_HEADER)

    # save the plot to a png file for later review
    png = '{}-{}-{}.Deploy{:02d}.review_plots.png'.format(site, node, sensor, deployment)
    fig.savefig(os.path.join(out_path, png), format='png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
