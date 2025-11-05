#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Load the CTDBP data from the uncabled, Coastal Endurance Surface
    Moorings and processes the data to generate report plots and tables for
    HITL QC review.
"""
import dateutil.parser as parser
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import warnings

from ooi_data_explorations.common import get_annotations, get_deployment_dates, load_gc_thredds
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.uncabled.process_nutnr import suna_datalogger, suna_instrument
from ooi_data_explorations.qartod.discrete_samples import get_discrete_samples, distance_to_cast
from ooi_data_explorations.qartod.qc_processing import ANNO_HEADER, inputs
from ooi_data_explorations.qartod.reporting import apply_qc_results


def combine_delivery_methods(site, node, sensor, deployment):
    """
    Downloads the  data from each of the three data delivery methods for
    the uncabled SUNA (NUTNR) instrument for a specific deployment and
    returns a list object containing the data from each delivery method.

    :param site: Site designator, extracted from the first part of the
        reference designator
    :param node: Node designator, extracted from the second part of the
        reference designator
    :param sensor: Sensor designator, extracted from the third and fourth part
        of the reference designator
    :param deployment: deployment number to download the data for
    :return merged: the three CTDBP data streams as a list object
    """
    # set the regex tag for the data files to download
    tag = 'deployment{:04d}.*NUTNR.*\\.nc$'.format(deployment)
    print('### Downloading data for {}-{}-{}, deployment {:02d} ###'.format(site, node, sensor, deployment))
    warnings.filterwarnings("error")  # set warnings to raise exceptions so we can catch them
    # download the data from the telemetered data stream
    try:
        telem = load_gc_thredds(site, node, sensor, 'telemetered', 'suna_dcl_recovered', tag)
    except UserWarning:
        telem = None
    # download the data from the recovered_host data stream
    try:
        rhost = load_gc_thredds(site, node, sensor, 'recovered_host', 'suna_dcl_recovered', tag)
    except UserWarning:
        rhost = None
    # download the data from the recovered_inst data stream
    try:
        rinst = load_gc_thredds(site, node, sensor, 'recovered_inst', 'suna_instrument_recovered', tag)
    except UserWarning:
        rinst = None
    # reset the warnings to default behavior
    warnings.resetwarnings()

    # combine the data from the three delivery methods into a list object
    return [telem, rhost, rinst]


def plotting(site, node, sensor, deployment, dates, availability, merged, qartod, pre_merged, post_merged, samples):
    """
    Generate plots of the CTDBP data for the telemetered, recovered host and
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

    # create the plots for each parameter
    ax1 = fig.add_subplot(gs[0, :])
    labels = ['T', 'RH', 'RI']
    colors = ['RoyalBlue', 'DarkOrange', 'ForestGreen']
    yvalues = [1.3, 1, 0.7]
    for i in range(len(availability)):
        if len(availability[i]) > 0:
            ax1.plot(availability[i], availability[i].astype(int) * 0 + yvalues[i], 'o',
                     label=labels[i], color=colors[i])
    ax1.set_xlim(dates)
    ax1.set_ylim([0.5, 1.50])
    ax1.set_yticks(yvalues)
    ax1.set_yticklabels(labels)

    ax2 = fig.add_subplot(gs[1:3, :])
    ax2.plot(merged.time, merged.nitrate_concentration, label='reported', color='DarkOrange')
    ax2.plot(merged.time, merged.corrected_nitrate_concentration, label='calculated', color='RoyalBlue')
    ax2.plot(dates, [-2, -2], label='', color='DarkRed')
    ax2.set_xlim(dates)
    ax2.set_ylim([-5, 45])
    ax2.set_yticks([-5, 0, 15, 30, 45])
    ax2.set_ylabel('Nitrate (umol/L)')
    ax2.legend(loc='upper left')

    ax3 = fig.add_subplot(gs[3:5, 0])
    zoom = [dates[0] - pd.Timedelta('15D'), dates[0] + pd.Timedelta('15D')]
    ax3.plot(merged.time, merged.nitrate_concentration, label='reported', color='DarkOrange')
    ax3.plot(merged.time, merged.corrected_nitrate_concentration, label='calculated', color='RoyalBlue')
    ax3.plot(zoom, [-2, -2], label='', color='DarkRed')
    if pre_merged is not None:
        ax3.plot(pre_merged.time, pre_merged.corrected_nitrate_concentration, label='Pre', color='grey')
    ax3.scatter(samples['Start Time [UTC]'], samples['Discrete Nitrate [uM]'], marker='*', label='Discrete',
                color='ForestGreen')
    ax3.set_xlim(zoom)
    x_fmt = mdates.DateFormatter('%b-%d')
    ax3.xaxis.set_major_formatter(x_fmt)
    ax3.set_xlabel('')
    ax3.set_ylim([-5, 45])
    ax3.set_yticks([-5, 0, 15, 30, 45])
    ax3.set_ylabel('Nitrate (umol/L)')
    ax3.legend(loc='upper left')

    ax4 = fig.add_subplot(gs[3:5, 1])
    zoom = [dates[1] - pd.Timedelta('15D'), dates[1] + pd.Timedelta('15D')]
    ax4.plot(merged.time, merged.nitrate_concentration, label='reported', color='DarkOrange')
    ax4.plot(merged.time, merged.corrected_nitrate_concentration, label='calculated', color='RoyalBlue')
    ax4.plot(zoom, [-2, -2], label='', color='DarkRed')
    if post_merged is not None:
        ax4.plot(post_merged.time, post_merged.corrected_nitrate_concentration, label='Post', color='grey')
    ax4.scatter(samples['Start Time [UTC]'], samples['Discrete Nitrate [uM]'], marker='*', label='Discrete',
                color='ForestGreen')
    ax4.set_xlim(zoom)
    ax4.xaxis.set_major_formatter(x_fmt)
    ax4.set_xlabel('')
    ax4.set_ylim([-5, 45])
    ax4.set_yticks([-5, 0, 15, 30, 45])
    ax4.legend(loc='upper left')

    ax5 = fig.add_subplot(gs[5, :])
    ax5.plot(merged.time, merged.spectrum_average, label='Spectrum', color='RoyalBlue')
    ax5.plot(merged.time, merged.dark_value_used_for_fit, label='Dark', color='DarkOrange')
    ax5.plot(dates, [10000, 10000], label='', color='DarkRed')
    ax5.set_xlim(dates)
    ax5.set_ylim([0, 50000])
    ax5.set_ylabel('Average (counts)')
    ax5.legend(loc='upper left')

    ax6 = fig.add_subplot(gs[6, :])
    ax6.plot(merged.time, merged.absorbance_at_254_nm, label='254 nm', color='RoyalBlue')
    ax6.plot(merged.time, merged.absorbance_at_350_nm, label='350 nm', color='DarkOrange')
    ax6.plot(dates, [1.3, 1.3], label='', color='DarkRed')
    ax6.set_xlim(dates)
    ax6.set_ylim([0, 5])
    ax6.set_ylabel('Absorption (AU)')
    ax6.legend(loc='upper left')

    ax7 = fig.add_subplot(gs[7, :])
    ax7.plot(merged.time, merged.fit_rmse, label='Measured', color='RoyalBlue')
    ax7.plot(dates, [0.001, 0.001], label='Suspect', color='DarkRed')
    ax7.set_xlim(dates)
    ax7.set_ylim([1e-6, 1.0])
    ax7.set_yticks([1e-6, 1e-4, 1e-2, 1.0])
    ax7.set_yscale('log')
    ax7.set_ylabel('RMSE')

    ax8 = fig.add_subplot(gs[8, :])
    new = qartod[0]
    old = qartod[1]
    anno = qartod[2]
    self = qartod[3]
    if len(qartod) > 0:
        ax8.plot(new[0].time, new[0].values, '*', label='QARTOD', color='RoyalBlue')
    if len(old) > 0:
        ax8.plot(old[0].time, old[0].values, '.', label='1.5', color='DarkOrange')
    if len(self) > 0:
        ax8.plot(self[0].time, self[0].values, '+', label='Self', color='ForestGreen')
    if len(anno) > 0:
        ax8.plot(anno[0].time, anno[0].values, 'o', label='HITL', color='DarkRed')

    ax8.set_xlim(dates)
    ax8.set_ylim([0, 5])
    ax8.set_yticks([0, 1, 2, 3, 4])
    ax8.legend(loc='upper left')

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

    :return merged: the merged NUTNR data streams for the deployment
    :return fig: the report figure object for the deployment
    :return annotations: the annotations for the sensor for the deployment
    """
    # load the data from the three delivery methods
    data = combine_delivery_methods(site, node, sensor, deployment)

    # also grab the data from the prior and post deployments
    pre_data = combine_delivery_methods(site, node, sensor, deployment - 1)
    post_data = combine_delivery_methods(site, node, sensor, deployment + 1)

    # create an availability data array for each data stream
    availability = [d.time for d in data if d is not None]

    # get the start and end dates for the deployment
    start, end = get_deployment_dates(site, node, sensor, deployment)
    start = parser.parse(start, ignoretz=True)
    end = parser.parse(end, ignoretz=True)
    dates = [start, end]

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
                data[i] = suna_datalogger(data[i])
                data[i] = apply_qc_results(data[i], annotations)
            if pre_data[i]:
                pre_data[i] = suna_datalogger(pre_data[i])
                pre_data[i] = apply_qc_results(pre_data[i], annotations)
            if post_data[i]:
                post_data[i] = suna_datalogger(post_data[i])
                post_data[i] = apply_qc_results(post_data[i], annotations)
        else:
            if data[i]:
                data[i] = suna_instrument(data[i])
                data[i] = apply_qc_results(data[i], annotations)
            if pre_data[i]:
                pre_data[i] = suna_instrument(pre_data[i])
                pre_data[i] = apply_qc_results(pre_data[i], annotations)
            if post_data[i]:
                post_data[i] = suna_instrument(post_data[i])
                post_data[i] = apply_qc_results(post_data[i], annotations)

    # combine the data from the three delivery methods into a single dataset
    merged = combine_datasets(data[0], data[1], data[2], None)

    # pull out the QC results for the parameters of interest
    qartod = [
        [merged[x] for x in merged.variables if 'qartod_results' in x],
        [merged[x] for x in merged.variables if 'qc_summary_results' in x],
        [merged[x] for x in merged.variables if 'annotations_qc_results' in x],
        [merged[x] for x in merged.variables if 'quality_flag' in x],
    ]

    # now resample the data to hourly intervals (adding a 30-minute offset to the time to center the data)
    merged['time'] = merged['time'] + pd.Timedelta('30Min')
    merged = merged.resample(time='1h', skipna=True).median(dim='time', keep_attrs=True)
    merged = merged.where(~np.isnan(merged.deployment), drop=True)

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
    samples = samples[(samples['CTD Depth [m]'] - 7.0).abs() <= 3.0]
    samples = samples.reset_index()

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

    # return the data, the figure object and the annotations
    return merged, fig, annotations


def main(argv=None):
    """
    Download the CTDBP data from the Gold Copy THREDDS server and create a
    plot page and annotations table for subsequent HITL QC reviews.
    """
    # set up the input arguments
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    deployment = args.deployment

    # create a local directory to store the data and plots
    out_path = os.path.join(os.path.expanduser('~'), 'ooidata/reports')
    out_path = os.path.join(out_path, site, 'midwater', 'nutnr')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # generate the report data, plot and annotations
    data, fig, annotations = generate_report(site, node, sensor, deployment)

    # save the data to a netCDF file for further processing
    nc_file = '{}-{}-{}.Deploy{:02d}.merged_data.nc'.format(site, node, sensor, deployment)
    data.to_netcdf(os.path.join(out_path, nc_file))

    # save the annotations to a csv file for further processing
    anno_csv = '{}-{}-{}.Deploy{:02d}.quality_annotations.csv'.format(site, node, sensor, deployment)
    annotations.to_csv(os.path.join(out_path, anno_csv), index=False, columns=ANNO_HEADER)

    # save the plot to a png file for later review
    png = '{}-{}-{}.Deploy{:02d}.review_plots.png'.format(site, node, sensor, deployment)
    fig.savefig(os.path.join(out_path, png), format='png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
