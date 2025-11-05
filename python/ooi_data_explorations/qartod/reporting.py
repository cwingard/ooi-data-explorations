#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief Utility functions for creating data reports.
"""
import os.path

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from ooi_data_explorations.common import add_annotation_qc_flags
from ooi_data_explorations.qartod.qc_processing import ANNO_HEADER


def apply_qc_results(ds, annotations):
    """
    Use the annotations, any variables with QARTOD tests applied, and any
    variables with instrument specific QC tests applied to NaN values that were
    marked as fail in order to exclude them from further analysis.

    :param ds: xarray dataset containing the data to be cleaned
    :param annotations: dictionary containing the annotations for the data set
    :return ds: cleaned xarray dataset
    """
    # create a list of any variables that have had QARTOD tests applied
    variables = [x.split('_qartod_results')[0] for x in ds.variables if 'qartod_results' in x]

    # if we have any tests results, NaN out the ones that failed
    if variables:
        for v in variables:
            ds[v] = ds[v].where(ds[v + '_qartod_results'] != 4)

    # create a list of any variables that have had the older OOI QC tests applied
    variables = [x.split('_qc_summary_flag')[0] for x in ds.variables if '_qc_summary_flag' in x]

    # if we have any tests results, NaN out the ones that failed
    if variables:
        for v in variables:
            ds[v] = ds[v].where(ds[v + '_qc_summary_flag'] != 4)

    # create a list of any variables that have had instrument specific tests applied
    variables = [x.split('_quality_flag')[0] for x in ds.variables if '_quality_flag' in x]

    # if we have any tests results, NaN out the ones that failed
    if variables:
        if len(variables) > 1:
            for v in variables:
                ds[v] = ds[v].where(ds[v + '_quality_flag'] != 4)
        else:
            ds = ds.where(ds[variables[0] + '_quality_flag'] != 4)

    # now add the annotations to the data set
    if not annotations.empty:
        ds = add_annotation_qc_flags(ds, annotations)

        # create a list of any variables that have had individual annotation flags assigned
        variables = [x.split('_annotations_qc_results')[0] for x in ds.variables if '_annotations_qc_results' in x]

        # if we have any variable specific HITL annotations, NaN out the ones that failed
        if variables:
            for v in variables:
                if v in ds.variables:
                    ds[v] = ds[v].where(ds[v + '_annotations_qc_results'] != 4)

        # finally, remove data where the rollup annotation is set to fail
        if 'rollup_annotations_qc_results' in ds.variables:
            ds = ds.where(ds['rollup_annotations_qc_results'] != 4, drop=True)

    # return the cleaned-up record with annotations (if any, added)
    return ds


def create_pdf(fig, annotations, report):
    """
    Create a PDF report containing the figure and any annotations.

    :param fig: matplotlib figure to include in the report
    :param annotations: dictionary containing the annotations for the data set
    :param report: path to the PDF file to create
    """
    with PdfPages(report) as pdf:
        # add the figure to the PDF
        pdf.savefig(fig)
        plt.close(fig)

        # create an annotations table
        tab, ax = plt.subplots(figsize=(17, 11))
        ax.axis('off')
        ax.table(cellText=annotations.values, colLabels=annotations.columns, loc='center')
        pdf.savefig(tab)
        plt.close(tab)
