#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from ast import literal_eval
import dateutil.parser as parser
import numpy as np
import os
import pandas as pd
import pytz
import s3fs
import sys
import xarray as xr

from ooi_data_explorations.common import get_annotations, add_annotation_qc_flags
from ooi_data_explorations.qartod.qc_processing import process_gross_range, process_climatology
import decimate


def decimateData(xs,decimationThreshold):
    xs = xs.dropna('time')
    # decimate data
    dec_data_df = decimate.downsample(xs, decimationThreshold)
    # turn dataframe into dataset
    dec_data = xr.Dataset.from_dataframe(dec_data_df, sparse=False)
    dec_data = dec_data.swap_dims({'index': 'time'})
    dec_data = dec_data.reset_coords()
    dec_data.attrs = xs.attrs

    return dec_data


def exportTables(qartodDict,site,node,sensor,qartod_tests):

    headers = {}
    headers['gross_range'] = ['subsite', 'node', 'sensor', 'stream', 'parameter', 'qcConfig', 'source', 'notes']
    headers['climatology'] = ['subsite', 'node', 'sensor', 'stream', 'parameters', 'climatologyTable', 'source', 'notes']
    testOrder = {'lookup': 0,'table': 1}

    folderPath = os.path.join(os.path.expanduser('~'), 'qartod_staging')
    folderPath = os.path.abspath(folderPath)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    for param in qartodDict:
        for test in qartodDict[param]:
            output = qartod_tests[test]['output']
            print('output: ', output)
            for fileOut in output:
                print('fileOut ', fileOut)
                if 'lookup' in fileOut:
                    for method in qartodDict[param][test]:
                        print('method: ', method)
                        qartod_csv_name = '-'.join([site,node,sensor,param]) + '.' + test + '.csv' + '.' + method
                        qartod_csv = os.path.join(folderPath, qartod_csv_name)
                        print('exporting ',qartod_csv)
                        if len(output) > 1:
                            if 'binned' in method:
                                with open(qartod_csv, 'w') as lkup:
                                    for key in qartodDict[param][test][method]:
                                        lkup.write(str(key))
                                        lkup.write(",")
                                        qartodRow = qartodDict[param][test][method][key][testOrder[fileOut]]
                                        if isinstance(qartodRow, str):
                                            lkup.write(qartodRow)
                                        else:
                                            lkup.write(qartodRow.to_string())
                                        lkup.write("\n")
                            else:
                                qartodOut = qartodDict[param][test][method][testOrder[fileOut]]
                                qartodOut.to_csv(qartod_csv, index=False, columns = headers[test])
                        else:
                            if 'binned' in method:
                                with open(qartod_csv, 'w') as lkup:
                                    for key in qartodDict[param][test][method]:
                                        lkup.write(str(key))
                                        lkup.write(",")
                                        qartodRow = qartodDict[param][test][method][key]
                                        if isinstance(qartodRow, str):
                                            lkup.write(qartodRow)
                                        else:
                                            lkup.write(qartodRow.to_string())
                                        lkup.write("\n")
                            else:
                                qartodOut = qartodDict[param][test][method]
                                qartodOut.to_csv(qartod_csv, index=False, columns = headers[test])

                elif 'table' in fileOut:
                    for method in qartodDict[param][test]:
                        print('method: ', method)
                        qartod_csv_name = '-'.join([site,node,sensor,param]) + '.' + test + '.csv' + '.' + method
                        qartod_csv = os.path.join(folderPath, qartod_csv_name)
                        print('exporting ', qartod_csv)
                        if len(output) > 1:
                            if 'binned' in method:
                                with open(qartod_csv, 'w') as tbl:
                                    for key in qartodDict[param][test][method]:
                                        tbl.write(str(key))
                                        tbl.write(",")
                                        qartodRow = qartodDict[param][test][method][key][testOrder[fileOut]]
                                        tbl.write(','.join(qartodRow))
                                        tbl.write("\n")
                            else:
                                qartodOut = qartodDict[param][test][method][testOrder[fileOut]]
                                with open(qartod_csv, 'w') as tbl:
                                    for qartodRow in qartodOut:   
                                        tbl.write(qartodRow)

                        else:
                            if 'binned' in method:
                                with open(qartod_csv, 'w') as tbl:
                                    for key in qartodDict[param][test][method]:
                                        tbl.write(str(key))
                                        tbl.write(",")
                                        qartodRow = qartodDict[param][test][method][key]
                                        tbl.write(qartodRow)
                                        tbl.write("\n")
                            else:
                                qartodOut = qartodDict[param][test][method]
                                with open(qartod_csv, 'w') as tbl:
                                    for qartodRow in qartodOut:
                                        tbl.write(qartodRow)
    
    return

def filterData(data, node, site, sensor, param, cut_off):

    index = 1

    # get the current system annotations for the sensor
    annotations = get_annotations(site, node, sensor)
    annotations = pd.DataFrame(annotations)
    if not annotations.empty:
        annotations = annotations.drop(columns=['@class'])
        annotations['beginDate'] = pd.to_datetime(annotations.beginDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
        annotations['endDate'] = pd.to_datetime(annotations.endDT, unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

    # create an annotation-based quality flag
    data = add_annotation_qc_flags(data, annotations)

    # clean-up the data, NaN-ing values that were marked as fail in the QC checks and/or identified as a block
    # of failed data, and then removing all records where the rollup annotation (every parameter fails) was
    # set to fail.
    qcVar_summary_string = param + '_qc_summary_flag'
    if 'qcVar_sumamry_string' in data.variables:
        m = data[qcVar_summary_string] == 4
        data[param][m] = np.nan
    qcVar_results_string = param + '_qc_results'
    if qcVar_results_string in data.variables:
        m = data[qcVar_results_string] == 4
        data[param][m] = np.nan

    if 'rollup_annotations_qc_results' in data.variables:
        data = data.where(data.rollup_annotations_qc_results < 4)
 
    annotations_flag_string = param + '_annotations_qc_results'
    if annotations_flag_string in data.variables:
        data = data.where(data[annotations_flag_string] < 3)

    # if a cut_off date was used, limit data to all data collected up to the cut_off date.
    # otherwise, set the limit to the range of the downloaded data.
    if cut_off:
        cut = parser.parse(cut_off)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
        src_date = cut.strftime('%Y-%m-%d')
    else:
        cut = parser.parse(data.time_coverage_end)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
        src_date = cut.strftime('%Y-%m-%d')

    data = data.sel(time=slice('2014-01-01T00:00:00', end_date))
    #####data = data.sel(time=slice('2016-01-15T00:00:00', end_date))

    return data, annotations


def inputs(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # initialize argument parser
    inputParser = argparse.ArgumentParser(
        description="""Download and process instrument data to generate QARTOD lookup tables""")

    # assign input arguments.
    inputParser.add_argument("-rd", "--refDes", dest="refDes", type=str, required=True)
    inputParser.add_argument("-co", "--cut_off", dest="cut_off", type=str, required=False)
    inputParser.add_argument("-d", "--decThreshold", dest="decThreshold", type=str, required=True)
    inputParser.add_argument("-v", "--userVars", dest="userVars", type=str, required=True)

    # parse the input arguments and create a parser object
    args = inputParser.parse_args(argv)
 
    return args


def loadData(zarrDir):
    fs = s3fs.S3FileSystem(anon=True)
    zarr_store = fs.get_mapper('ooi-data/' + zarrDir)
    ds = xr.open_zarr(zarr_store, consolidated=True)

    return ds


def processData(data,param):
    print('in the processData loop: ', param)

    return data


def runQartod(test,data,param,limits,site,node,sensor,stream):

    if 'gross_range' in test:
        qartodResults = process_gross_range(data.compute(), [param], limits, site=site,
                                        node=node, sensor=sensor, stream=stream)

    elif 'climatology' in test:
        clm_lookup, clm_table = process_climatology(data.compute(), [param], limits, site=site,
                                                    depth_bins = np.array([]), node=node,
                                                    sensor=sensor, stream=stream)
        qartodResults = [clm_lookup, clm_table]
        

    return qartodResults


def main(argv=None):
    # setup input arguments
    args = inputs(argv)
    refDes = args.refDes
    cut_off = args.cut_off
    decThreshold = int(args.decThreshold)
    userVars = args.userVars

    # load parameter dictionaries
    param_dict = (pd.read_csv('parameterMap.csv',converters={"variables": literal_eval,"limits": literal_eval})).set_index('dataParameter').T.to_dict()

    sites_dict = (pd.read_csv('siteParameters.csv',converters={"variables": literal_eval})).set_index('refDes').T.to_dict('series')

    qartod_tests = (pd.read_csv('qartodTests.csv', converters={"output": literal_eval,"parameters": literal_eval,"profileCalc": literal_eval})).set_index('qartodTest').T.to_dict()

    platform = sites_dict[refDes]['platformType']

    # define sub-variables
    site, node, port, instrument,method,stream = sites_dict[refDes]['zarrFile'].split('-')
    sensor = port + '-' + instrument

    # load data
    data = loadData(sites_dict[refDes]['zarrFile'])
    allVars = list(data.keys())
 
    if 'all' in userVars:
        dataVars=sites_dict[refDes]['variables']
    else:
        dataVars = [userVars]
    
    paramList = []
    qartodTests_dict = {}
    for qcVar in dataVars:
        qartodTests_dict[qcVar] = {}
        qcParam = [i for i in param_dict if qcVar in param_dict[i]['variables']][0]
        qartodTests_dict[qcVar]['tests'] = {t for t in qartod_tests if qcParam in qartod_tests[t]['parameters']}
        qartodTests_dict[qcVar]['limits'] = param_dict[qcParam]['limits']
        for p in param_dict[qcParam]['variables']:
            paramList.append(p)
     
    if 'profiler' in platform:
        if 'int_ctd_pressure' in data:
            paramList.append('int_ctd_pressure')
        elif 'sea_water_pressure' in data:
            paramList.append('sea_water_pressure')

    dropList = [item for item in allVars if item not in paramList]
    data = data.drop_vars(dropList)
    qartodDict = {}
    
    if 'fixed' in platform:
        if ( (len(data['time']) > decThreshold) and (decThreshold > 0) ):
            data = decimateData(data, decThreshold)
        for param in dataVars:
            data = processData(data,param)
            data, annotations = filterData(data, node, site, sensor, param, cut_off)
            qartodDict[param] = {}
            for test in qartodTests_dict[param]['tests']:
                qartodDict[param][test] = {}
                qartodDict[param][test][platform] = runQartod(test,data,param,qartodTests_dict[param]['limits'],site,node,sensor,stream)

    elif 'profiler' in platform:
        if 'sea_water_pressure' in data:
            pressParam = 'sea_water_pressure'
        elif 'int_ctd_pressure' in data:   
            pressParam = 'int_ctd_pressure'
        else:
            print('no pressure parameter found; unable to bin data!')
        if 'SF0' in node:
            shallow_upper = np.arange(6,105,1)  
            shallow_lower = np.arange(105,200,5)
            binList = np.concatenate((shallow_upper,shallow_lower), axis=0).tolist()
        elif 'DP0' in node:
            maxDepth = {'DP01A': 2900, 'DP01B': 600, 'DP03A': 2600}
            binList = np.arange(200,maxDepth[node], 5).tolist()
        bins = []
        for i in range(0,len(binList)-1):
            bins.append((binList[i], binList[i+1]))

        for param in dataVars:
            qartodDict[param] = {}
            for test in qartodTests_dict[param]['tests']:
                qartodDict[param][test] = {}
                if 'integrated' in qartod_tests[test]['profileCalc']:
                    if ( (len(data['time']) > decThreshold) and (decThreshold > 0) ): 
                        data = decimateData(data, decThreshold)
                    data = processData(data,param)
                    data, annotations = filterData(data, node, site, sensor, param, cut_off)
                    qartodDict[param][test]['int'] = runQartod(test,data,param,qartodTests_dict[param]['limits'],site,node,sensor,stream)
                
                if 'binned' in qartod_tests[test]['profileCalc']:
                    qartodDict[param][test]['binned'] = {}
                    for pressBin in bins:
                        print('pressBin: ', pressBin)
                        data_bin = data.where( (pressBin[0] < data[pressParam].compute()) & (data[pressParam].compute() < pressBin[1]), drop=True )
            
                        if (data_bin[pressParam].isnull()).all():
                            print('no data available for bin: ', pressBin)
                        else:
                            if ( (len(data['time']) > decThreshold) and (decThreshold > 0) ):
                                data_bin = decimateData(data_bin, decThreshold)
                            data_bin = processData(data_bin,param)
                            data_bin, annotations = filterData(data_bin, node, site, sensor, param, cut_off)
                            try:
                                print('trying ', test)
                                qartodRow = runQartod(test,data_bin,param,qartodTests_dict[param]['limits'],site,node,sensor,stream)
                            except:
                                print('failed runQartod for ', test)
                                qartodRow = 'unable to calculate for pressure bin'
                            qartodDict[param][test]['binned'][pressBin] = qartodRow
                       
    exportTables(qartodDict,site,node,sensor,qartod_tests)    
    

if __name__ == '__main__':
    main()
