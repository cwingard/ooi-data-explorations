#!/usr/bin/env bash
#
# harvest_ce_ctdpf.sh
#
# Harvest the ctdpf data from all of the OOI Coastal Endurance profiler
# moorings. Data sets are limited to the recovered profiler data for the CSPP
# and the telemetered and recovered data for the WFP. Data is downloaded from
# OOI Net and organized by the mooring, mooring sub-location, data delivery
# method and deployment.
#
# C. Wingard, 2019-07-22 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.data_request"
ASSEMBLY="profiler"
INSTRMT="ctdpf"

### CE01ISSP ###
BASE_FLAGS="-s CE01ISSP -a $ASSEMBLY -i $INSTRMT"
BASE_FILE="${HOME}/ooidata/m2m/ce01issp/profiler/ctdpf/ce01issp.cspp.ctdpf"
for i in $(seq -f "%02g" 1 15); do
    $PYTHON $BASE_FLAGS -m recovered_cspp -dp $i -o "$BASE_FILE.deploy$i.recovered_cspp.ctdpf_j_cspp_instrument_recovered.nc"
done

### CE02SHSP ###
BASE_FLAGS="-s CE02SHSP -a $ASSEMBLY -i $INSTRMT"
BASE_FILE="${HOME}/ooidata/m2m/ce02shsp/profiler/ctdpf/ce02shsp.cspp.ctdpf"
for i in $(seq -f "%02g" 1 18); do
    $PYTHON $BASE_FLAGS -m recovered_cspp -dp $i -o "$BASE_FILE.deploy$i.recovered_cspp.ctdpf_j_cspp_instrument_recovered.nc"
done

### CE06ISSP ###
BASE_FLAGS="-s CE06ISSP -a $ASSEMBLY -i $INSTRMT"
BASE_FILE="${HOME}/ooidata/m2m/ce06issp/profiler/ctdpf/ce06issp.cspp.ctdpf"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -m recovered_cspp -dp $i -o "$BASE_FILE.deploy$i.recovered_cspp.ctdpf_j_cspp_instrument_recovered.nc"
done

### CE07SHSP ###
BASE_FLAGS="-s CE07SHSP -a $ASSEMBLY -i $INSTRMT"
BASE_FILE="${HOME}/ooidata/m2m/ce07shsp/profiler/ctdpf/ce07shsp.cspp.ctdpf"
for i in $(seq -f "%02g" 1 9); do
    $PYTHON $BASE_FLAGS -m recovered_cspp -dp $i -o "$BASE_FILE.deploy$i.recovered_cspp.ctdpf_j_cspp_instrument_recovered.nc"
done
