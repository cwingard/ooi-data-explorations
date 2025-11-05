#!/usr/bin/env bash
#
# qartod_ce_ctdpf.sh
#
# Collect the ctdpf data from all of the OOI Coastal Endurance Profilers
# to calculate QARTOD test ranges and generate the different lookup
# values and tables.
#
# C. Wingard, 2022-06-17 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.qartod.endurance.qartod_ce_ctdpf"

### CE01ISSP ###
$PYTHON -s CE01ISSP -n SP001 -sn 05-CTDPFJ000 -co 2023-12-31T23:59:59

### CE02SHSP ###
$PYTHON -s CE02SHSP -n SP001 -sn 02-CTDPFJ000 -co 2023-12-31T23:59:59

### CE06ISSP ###
$PYTHON -s CE06ISSP -n SP001 -sn 05-CTDPFJ000 -co 2023-12-31T23:59:59

### CE07SHSP ###
$PYTHON -s CE07SHSP -n SP001 -sn 02-CTDPFJ000 -co 2023-12-31T23:59:59

### CE09OSPM ###
$PYTHON -s CE09OSPM -n WFP01 -sn 03-CTDPFK000 -co 2023-12-31T23:59:59
